# Frame-level expansion of text and datetime columns.
#
# Port of what TabPFN v3.5's estimator does before validation, in its order:
# `DateTransformer`, then `TextTransformer`. Durations become seconds in
# place; each point in time is swapped for its calendar features; each text
# column for its 30 LSA components. The column encoders themselves live in
# `R/prep-datetime.R` and `R/prep-text.R`; this file decides *which* columns
# they apply to, lays the result out as the reference does, and remembers
# enough to lay out new rows the same way.
#
# It runs on a data frame, ahead of `.encode_predictors()`, and never inside
# the member pipeline. Three things force that. `.encode_predictors()` maps
# one column to one column, so it cannot express a 1-to-30 expansion. The
# member pipeline receives a matrix, by which point a string is already a
# code. And its feature budget is a hard abort rather than a subsample, so
# expanded columns have to be counted by it, which means existing before it.
#
# Layout, as the reference's `_drop_and_append` produces it twice over: the
# columns that were not expanded, in their original order, then every date
# column's features in input order, then every text column's. A kept
# column's index therefore moves down by the number of expanded columns that
# preceded it -- the reason declared categorical indices are remapped.
#
# Behaviour with the flags off is deliberately *not* the reference's. Python
# refuses a datetime column when `TRANSFORM_DATES` is off; R has always
# turned one into a single number, and refusing now would break every
# existing caller on every other backend. So "off" keeps today's behaviour.

#' Does this string spell a number, as pandas reads one?
#'
#' `pd.to_numeric(errors = "coerce")`, which is how the reference tells a
#' column of numbers-stored-as-text from a text column. R's `as.numeric()`
#' agrees on nearly everything and disagrees on two spellings that matter:
#' it reads `"NaN"` as a number and pandas does not, and it reads hex
#' (`"0x10"`) and pandas does not.
#' @keywords internal
.is_numeric_spelling <- function(x) {
  v <- suppressWarnings(as.numeric(x))
  !is.na(v) & !grepl("^\\s*[+-]?0[xX]", x)
}

#' Which character columns count as text?
#'
#' The reference's rule, from `_text_positions`: a string column -- not a
#' factor, which is R's `category` -- with *more than* `min_cardinality`
#' distinct values counting a missing value as one of them, whose non-missing
#' values do not all spell numbers, and that the caller has not declared
#' categorical.
#'
#' Note the cutoff is 30, the same number as `max_unique` in
#' [detect_categorical_features()], and means the opposite: there, at most 30
#' levels is what a declared categorical may have; here, more than 30 is
#' what makes a string column text. They are not one threshold, and should
#' not be unified.
#'
#' @param data A data frame.
#' @param min_cardinality Distinct-value count above which a string column
#'   is text.
#' @param declared Names of columns declared categorical; never text.
#' @return Names of the text columns.
#' @keywords internal
.text_columns <- function(data, min_cardinality = 30L, declared = character()) {
  if (!is.data.frame(data)) return(character())
  nm <- names(data)
  keep <- vapply(seq_along(data), function(j) {
    col <- data[[j]]
    if (!is.character(col) || nm[[j]] %in% declared) return(FALSE)
    n_distinct <- length(unique(col))            # NA counts, as dropna=False
    if (n_distinct <= min_cardinality) return(FALSE)
    seen <- col[!is.na(col)]
    !all(.is_numeric_spelling(seen))
  }, logical(1))
  nm[keep]
}

#' Resolve a `transform_text` / `transform_dates` setting
#'
#' `"auto"` reads the checkpoint's own recipe: the `inference_config` the
#' v3.5 converter copies out of the safetensors header, which sets
#' `TRANSFORM_TEXT` and `TRANSFORM_DATES` on. Every other checkpoint carries
#' no such recipe, so `"auto"` is off for them and the feature is opt-in.
#' @keywords internal
.resolve_transform_flag <- function(flag, config, key, arg) {
  if (isTRUE(flag) || isFALSE(flag)) return(flag)
  if (identical(flag, "auto")) {
    return(isTRUE(config$inference_config[[key]]))
  }
  cli::cli_abort("{.arg {arg}} must be {.val auto}, {.val TRUE} or {.val FALSE}.")
}

# The preprocessing options, validated once. Carried on the model object so
# a later `fit()` on the engine API sees what its constructor was told.
# @keywords internal
.preprocess_options <- function(transform_text = "auto", transform_dates = "auto",
                                min_cardinality_for_text = 30L,
                                text_n_components = 30L) {
  chk <- function(v, arg) {
    if (!(isTRUE(v) || isFALSE(v) || identical(v, "auto"))) {
      cli::cli_abort("{.arg {arg}} must be {.val auto}, {.val TRUE} or {.val FALSE}.")
    }
    v
  }
  list(
    transform_text = chk(transform_text, "transform_text"),
    transform_dates = chk(transform_dates, "transform_dates"),
    min_cardinality_for_text = as.integer(min_cardinality_for_text),
    text_n_components = as.integer(text_n_components)
  )
}

# Does a frame have anything this file would touch? Answers without reading
# a model config, so an all-numeric frame pays nothing.
# @keywords internal
.frame_needs_expansion <- function(data, opts) {
  if (!is.data.frame(data)) return(FALSE)
  any(vapply(data, function(col) .is_instant(col) || .is_duration(col),
             logical(1))) ||
    length(.text_columns(data, opts$min_cardinality_for_text)) > 0L
}

#' Fit the frame expansion
#'
#' Decides, from the training rows alone, which columns are expanded and how
#' wide each becomes, and returns both the expanded frame and the state that
#' replays it. Everything is keyed by column name.
#'
#' @param data Predictor data frame.
#' @param transform_text,transform_dates Resolved logical flags.
#' @param min_cardinality_for_text,text_n_components See [tabfound()].
#' @param declared Names of columns declared categorical.
#' @return `list(data, state)`.
#' @keywords internal
fit_frame_expansion <- function(data, transform_text, transform_dates,
                                min_cardinality_for_text = 30L,
                                text_n_components = 30L,
                                declared = character()) {
  nm <- names(data)
  durations <- nm[vapply(data, .is_duration, logical(1))]
  instants <- nm[vapply(data, .is_instant, logical(1))]

  date_fits <- list()
  if (isTRUE(transform_dates) && length(instants)) {
    bad <- intersect(instants, declared)
    if (length(bad)) {
      cli::cli_abort(c(
        "Date column{?s} {.field {bad}} {?is/are} declared categorical.",
        i = "A date becomes several numeric features, so there is no single \\
             column the declaration could apply to. Drop it from \\
             {.arg categorical_features}, or turn it into a factor yourself."
      ))
    }
    date_fits <- stats::setNames(
      lapply(instants, function(n) fit_datetime_encoder(data[[n]], n)), instants)
  }

  text_cols <- if (isTRUE(transform_text)) {
    .text_columns(data, min_cardinality_for_text, declared)
  } else character()
  text_fits <- stats::setNames(
    lapply(text_cols, function(n) {
      fit_string_encoder(data[[n]], n, n_components = text_n_components, seed = 0L)
    }), text_cols)

  expanded <- c(names(date_fits), names(text_fits))
  # A string column that is not text is categorical, and gets the level set
  # `factor()` would have given it before mold -- recorded, so new rows are
  # coded against the training levels rather than their own.
  leftover_chr <- setdiff(nm[vapply(data, is.character, logical(1))], expanded)
  char_levels <- stats::setNames(
    lapply(leftover_chr, function(n) levels(factor(data[[n]]))), leftover_chr)

  state <- structure(
    list(input_names = nm, durations = durations, instants = instants,
         dates = date_fits, text = text_fits, char_levels = char_levels,
         transform_text = isTRUE(transform_text),
         transform_dates = isTRUE(transform_dates)),
    class = "tabfound_expansion"
  )
  out <- .assemble_expansion(data, state, fitting = TRUE)
  state$output_names <- names(out)
  list(data = out, state = state)
}

#' Replay a fitted frame expansion on new rows
#'
#' Refuses rather than guesses when the new rows do not match what was
#' fitted: a column expanded as text that now holds numbers, a date column
#' that is no longer a date or is in another timezone, or a column that has
#' become a date since.
#'
#' @return The expanded data frame, with the columns and names of the fit.
#' @keywords internal
transform_frame_expansion <- function(data, state) {
  missing_cols <- setdiff(c(names(state$dates), names(state$text)), names(data))
  if (length(missing_cols)) {
    cli::cli_abort(c(
      "{.arg newdata} is missing column{?s} the model expanded at fit: {.field {missing_cols}}."
    ))
  }
  if (state$transform_dates) {
    new_instants <- names(data)[vapply(data, .is_instant, logical(1))]
    surprise <- setdiff(intersect(new_instants, state$input_names), names(state$dates))
    if (length(surprise)) {
      cli::cli_abort(c(
        "Column{?s} {.field {surprise}} hold{?s/} dates now but did not when the model was fitted.",
        i = "Pass {?it/them} with the class {?it/they} had at fit."
      ))
    }
  }
  for (n in names(state$text)) {
    col <- data[[n]]
    if (is.numeric(col) && any(!is.na(col))) {
      cli::cli_abort(c(
        "Column {.field {n}} held text when the model was fitted but holds numbers now.",
        i = "A column encoded as text needs strings at predict too."
      ))
    }
  }
  out <- .assemble_expansion(data, state, fitting = FALSE)
  names(out) <- state$output_names
  out
}

# Lay out one frame under a state: convert durations in place, code
# leftover strings against their training levels, drop the expanded
# columns, and append their features -- dates first, then text.
# @keywords internal
.assemble_expansion <- function(data, state, fitting) {
  for (n in intersect(state$durations, names(data))) {
    if (.is_duration(data[[n]])) data[[n]] <- duration_seconds(data[[n]])
  }
  novel <- character()
  for (n in intersect(names(state$char_levels), names(data))) {
    col <- data[[n]]
    if (is.factor(col)) col <- as.character(col)
    if (!is.character(col)) next
    lv <- state$char_levels[[n]]
    if (!fitting) {
      seen <- setdiff(unique(col[!is.na(col)]), lv)
      if (length(seen)) {
        # A free-text column coded as categorical has a new "level" in nearly
        # every row, so list a few and count the rest rather than print them
        # all.
        shown <- utils::head(seen, 5L)
        more <- if (length(seen) > 5L) sprintf(" and %d more", length(seen) - 5L) else ""
        novel <- c(novel, sprintf("%s: %s%s", n, paste(shown, collapse = ", "), more))
      }
    }
    data[[n]] <- factor(col, levels = lv)
  }
  if (length(novel)) {
    n_cols <- length(novel)
    cli::cli_warn(c(
      "{.arg newdata} has values the model was not fitted on in \\
       {n_cols} column{?s}; they become {.val NA}.",
      set_names(novel, rep("*", n_cols)),
      i = if (!state$transform_text)
        "A string column with many distinct values is treated as text when \\
         {.code transform_text = TRUE}, which reads unseen values by the \\
         character n-grams they share with the training column."
    ))
  }

  expanded <- c(names(state$dates), names(state$text))
  kept <- data[setdiff(names(data), expanded)]
  blocks <- c(
    lapply(names(state$dates), function(n) {
      transform_datetime_encoder(data[[n]], state$dates[[n]])
    }),
    lapply(names(state$text), function(n) {
      transform_string_encoder(data[[n]], state$text[[n]])
    })
  )
  if (!length(blocks)) return(kept)
  feats <- as.data.frame(do.call(cbind, blocks), check.names = FALSE)
  # A generated name that collides with a kept column is made unique, as
  # the reference's `make_names_unique` does; replayed from the fit at
  # predict, so the two always agree.
  all_names <- make.unique(c(names(kept), names(feats)), sep = "_")
  out <- cbind(kept, feats)
  names(out) <- all_names
  out
}

# Where a set of declared columns, named, sits after expansion.
# @keywords internal
.expanded_indices_of <- function(names_declared, state) {
  idx <- match(names_declared, state$output_names)
  as.integer(idx[!is.na(idx)])
}

# Put the typed columns back that a formula blueprint flattened.
#
# `hardhat`'s formula blueprint runs a model frame, which turns a `Date`
# into a plain number and so loses exactly what the date encoder needs.
# Rows and order survive the mold (it uses `na.pass`), and a predictor that
# is a bare column reference keeps that column's name -- so those columns
# are taken back from the raw data by name. A transformed term, `log(x)` or
# `as.numeric(when)`, has no raw column of its name and is left as the
# formula computed it, which is what whoever wrote it asked for.
# @keywords internal
#
# Only dates, durations and the text candidates kept as strings through
# mold are restored. An ordinary string column was made a factor before
# mold and hardhat already checked its levels; taking it back from the raw
# data would check them a second time and warn twice.
.restore_typed_columns <- function(predictors, raw, names_to_restore = NULL,
                                   keep_character = character()) {
  if (!is.data.frame(raw) || nrow(raw) != nrow(predictors)) return(predictors)
  if (is.null(names_to_restore)) {
    names_to_restore <- intersect(names(predictors), names(raw))
    names_to_restore <- names_to_restore[vapply(names_to_restore, function(n) {
      col <- raw[[n]]
      .is_instant(col) || .is_duration(col) ||
        (is.character(col) && n %in% keep_character)
    }, logical(1))]
  }
  for (n in intersect(names_to_restore, names(predictors))) {
    predictors[[n]] <- raw[[n]]
  }
  attr(predictors, "restored") <- names_to_restore
  predictors
}

#' @export
print.tabfound_expansion <- function(x, ...) {
  cli::cli_text("{.strong Text and date expansion}")
  items <- c(
    if (length(x$dates)) c("*" = "dates: {.field {names(x$dates)}}"),
    if (length(x$text)) c("*" = "text ({vapply(x$text, `[[`, 1L, 'n_out')} components each): {.field {names(x$text)}}"),
    if (length(x$durations)) c("*" = "durations, as seconds: {.field {x$durations}}")
  )
  if (length(items)) cli::cli_bullets(items) else cli::cli_text("nothing expanded")
  invisible(x)
}

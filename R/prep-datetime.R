# Datetime and duration columns, as TabPFN v3.5's estimator expands them.
#
# Port of `tabpfn.preprocessing.datetimes.DateTransformer`'s per-column step,
# which is skrub's `DatetimeEncoder` configured as
#
#   DatetimeEncoder(resolution = "second", add_weekday = TRUE,
#                   add_day_of_year = TRUE, periodic_encoding = "circular")
#
# A point in time becomes a handful of calendar features: the year, the
# minute and second, seconds since the epoch and the day of the year as they
# are, and the month, day, hour and weekday as sine/cosine pairs, so that
# December sits next to January and 23:00 next to midnight. A duration
# becomes its length in seconds and nothing else.
#
# Three details decide whether an R port agrees with the reference, and none
# of them is in skrub's documentation:
#
# * **The non-periodic features are float32.** skrub casts them with
#   `to_float32`, so seconds since the epoch near 2024 are stored at a
#   resolution of 128 seconds -- two timestamps a minute apart can come out
#   identical. Reproduced with [.round_float32()], because a port more
#   precise than its reference is a port that disagrees with it.
# * **The `day` period is a fixed 30.** Not the length of the month: the
#   31st sits a little past a full turn, and February's end short of one.
# * **The width depends on the data, but only in one way.** skrub does not
#   drop invariant features in general. What it does is cap the resolution
#   at `day` when every value sits at midnight, so a date-only column never
#   gets hour, minute or second features. That is decided at fit and
#   replayed at predict, which is why the fit is kept rather than refitted.
#
# Reference: skrub `_datetime_encoder.py` (`DatetimeEncoder`,
# `_CircularEncoder`, `_get_dt_feature`) and tabpfn
# `preprocessing/datetimes.py`.

# The features a point in time can yield, in the order skrub extracts them.
# `resolution = "second"` stops the calendar levels at `second`; the three
# additions follow in the order the encoder appends them.
.DT_LEVELS <- c("year", "month", "day", "hour", "minute", "second")

# `_DEFAULT_ENCODING_PERIODS`. Only these four are encoded circularly; the
# rest are passed through as numbers.
.DT_PERIODS <- c(month = 12, day = 30, hour = 24, weekday = 7)

#' Is this column a point in time?
#'
#' `Date`, `POSIXct` and `POSIXlt`. A duration is not -- it is a length, and
#' is converted in place rather than expanded -- and neither is a string
#' that happens to look like a date, which the reference also leaves alone.
#' @keywords internal
.is_instant <- function(x) inherits(x, c("Date", "POSIXct", "POSIXlt"))

#' @keywords internal
.is_duration <- function(x) inherits(x, "difftime")

# The timezone a column's calendar features are read in, by name. A `Date`
# has none. A `POSIXct` with no `tzone` attribute is read in the session's
# zone, which is what `format()` and `as.POSIXlt()` do with it, so that is
# the zone recorded -- otherwise the same object would expand differently
# on a machine set to another zone and nothing would say so.
# @keywords internal
.instant_tz <- function(x) {
  if (inherits(x, "Date")) return(NA_character_)
  tz <- attr(x, "tzone")
  tz <- if (is.null(tz)) "" else tz[[1L]]
  if (!nzchar(tz)) tz <- Sys.timezone() %||% "UTC"
  tz
}

# Calendar components and seconds since the epoch for one column.
#
# A `Date` is a day with no time: its components are read in UTC, where
# its midnight is, and its epoch seconds are whole days -- the same numbers
# pandas gives a naive `datetime64` at midnight. A `POSIXct` is an instant:
# its components are read in its own zone and its epoch seconds are counted
# from the UTC epoch, as pandas does for a timezone-aware column.
# @keywords internal
.instant_parts <- function(x, tz) {
  if (inherits(x, "POSIXlt")) x <- as.POSIXct(x)
  if (inherits(x, "Date")) {
    lt <- as.POSIXlt(x, tz = "UTC")
    total <- as.numeric(unclass(x)) * 86400
  } else {
    lt <- as.POSIXlt(x, tz = tz)
    total <- as.numeric(unclass(x))
  }
  wday <- lt$wday
  list(
    year = lt$year + 1900,
    month = lt$mon + 1,
    day = lt$mday,
    hour = lt$hour,
    minute = lt$min,
    # pandas' `dt.second` is the whole second; the fraction belongs to
    # `microsecond`, which `resolution = "second"` never extracts.
    second = floor(lt$sec),
    total_seconds = total,
    # pandas counts Monday as 0 and skrub adds one; R's `wday` counts
    # Sunday as 0. Monday = 1 ... Sunday = 7 either way round.
    weekday = ifelse(wday == 0, 7, wday),
    day_of_year = lt$yday + 1,
    # Whether the value sits exactly at midnight, fraction of a second
    # included -- `col.dt.normalize() == col`.
    midnight = lt$hour == 0 & lt$min == 0 & lt$sec == 0
  )
}

#' Fit the datetime encoder on one column
#'
#' Decides which features the column yields, and so how wide its expansion
#' is. The decision is the reference's: a column whose every non-missing
#' value sits at midnight is a date, and gets no time-of-day features.
#'
#' @param x A `Date`, `POSIXct` or `POSIXlt` vector.
#' @param name The column's name, used to name the features.
#' @return A fit, to hand to [transform_datetime_encoder()].
#' @keywords internal
fit_datetime_encoder <- function(x, name) {
  if (!.is_instant(x)) {
    cli::cli_abort("Column {.val {name}} is not a date or date-time.")
  }
  tz <- .instant_tz(x)
  parts <- .instant_parts(x, tz)
  seen <- !is.na(parts$year)
  # `_is_date` looks at the non-missing values only, and an all-missing
  # column counts as a date: pandas' `.all()` over nothing is TRUE.
  date_only <- all(parts$midnight[seen])

  levels <- if (date_only) .DT_LEVELS[seq_len(match("day", .DT_LEVELS))]
            else .DT_LEVELS
  extracted <- c(levels, "total_seconds", "weekday", "day_of_year")
  periodic <- extracted[extracted %in% names(.DT_PERIODS)]
  plain <- setdiff(extracted, periodic)
  outputs <- c(
    paste0(name, "_", plain),
    as.vector(rbind(paste0(name, "_", periodic, "_circular_0"),
                    paste0(name, "_", periodic, "_circular_1")))
  )
  structure(
    list(name = name, tz = tz, date_only = date_only,
         plain = plain, periodic = periodic, outputs = outputs,
         kind = if (inherits(x, "Date")) "Date" else "POSIXct"),
    class = "tabfound_datetime_fit"
  )
}

#' Apply a fitted datetime encoder
#'
#' Replays the fit: the same features, in the same order, whatever the new
#' rows hold -- a date-only column fitted as such stays hour-less even if a
#' later value has a time. A missing value makes every feature of its row
#' missing, which is what the reference does after filling it with a
#' placeholder to keep the circular encoder happy.
#'
#' @param x The column at predict time.
#' @param fit From [fit_datetime_encoder()].
#' @return A numeric matrix, one column per feature, named as at fit.
#' @keywords internal
transform_datetime_encoder <- function(x, fit) {
  if (!.is_instant(x)) {
    cli::cli_abort(c(
      "Column {.val {fit$name}} held dates when the model was fitted but does not now.",
      i = "Pass it with the class it had at fit ({.cls {fit$kind}})."
    ))
  }
  kind <- if (inherits(x, "Date")) "Date" else "POSIXct"
  if (!identical(kind, fit$kind)) {
    # A day and an instant are different things in R, as they are not in
    # pandas: a `Date` has no zone to read its calendar in, so there is no
    # faithful way to line one up against a column fitted as the other.
    cli::cli_abort(c(
      "Column {.val {fit$name}} was a {.cls {fit$kind}} when the model was \\
       fitted and is a {.cls {kind}} now.",
      i = "Pass it with the class it had at fit."
    ))
  }
  tz <- .instant_tz(x)
  if (!identical(tz, fit$tz)) {
    cli::cli_abort(c(
      "Column {.val {fit$name}} is in timezone {.val {tz}}, but was fitted in {.val {fit$tz}}.",
      i = "Calendar features are read in the column's own zone, so the same \\
           instant would expand differently. Convert it back with \\
           {.code attr(x, \"tzone\") <- {deparse(fit$tz)}}."
    ))
  }
  parts <- .instant_parts(x, fit$tz)
  missing <- is.na(parts$year)

  # Plain features pass through, as float32.
  plain <- lapply(fit$plain, function(f) .round_float32(parts[[f]]))
  # Periodic ones become a sine/cosine pair, in float64, computed in the
  # reference's order -- `X / period * 2 * pi` -- so the last bit agrees.
  circular <- unlist(lapply(fit$periodic, function(f) {
    v <- parts[[f]]
    v[missing] <- 0
    p <- .DT_PERIODS[[f]]
    list(sin(v / p * 2 * pi), cos(v / p * 2 * pi))
  }), recursive = FALSE)

  out <- do.call(cbind, c(plain, circular))
  out <- matrix(out, nrow = length(missing), dimnames = list(NULL, fit$outputs))
  out[missing, ] <- NA_real_
  out
}

#' A duration's length in seconds
#'
#' Always, and with no flag: the reference converts `timedelta64` in place
#' whether or not dates are expanded. It also fixes a quieter problem --
#' `as.numeric()` on a `difftime` returns the magnitude in whatever unit the
#' object happens to carry, so the same duration came out as 1.5 or 90
#' depending on how it was built.
#' @keywords internal
duration_seconds <- function(x) as.numeric(x, units = "secs")

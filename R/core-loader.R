# Backend-agnostic state_dict loader.
#
# Copies every tensor named in `state_dict_keys` into the matching
# parameter or buffer of a constructed nn_module. The only model-specific
# piece is `translate_fn`, which rewrites a checkpoint key into the R
# module path — R torch's `nn_module_list` inserts an extra `steps`
# component that PyTorch's `nn.Sequential` does not, and backends differ
# in how they wrap things.
#
# The loader is deliberately strict: an unmatched key, a shape mismatch,
# or an unfilled parameter is a bug in the backend definition, not
# something to paper over, because a silently unfilled tensor produces
# plausible-looking but wrong predictions.

#' Load checkpoint tensors into a constructed model
#'
#' @param model An `nn_module`.
#' @param weights Named list of tensors (from [read_safetensors()]).
#' @param keys Character vector of checkpoint keys to consume, in order.
#'   Usually `config$state_dict_keys`.
#' @param translate_fn Function mapping a checkpoint key to an R
#'   `named_parameters()` / `named_buffers()` path. Defaults to identity.
#' @param strict Logical. When `TRUE` (default), abort if any R-side
#'   parameter was left unfilled. When `FALSE`, warn instead.
#' @return `invisible(model)`.
#' @keywords internal
load_state_dict <- function(model, weights, keys,
                            translate_fn = identity,
                            strict = TRUE) {
  all_params  <- model$parameters
  all_buffers <- model$buffers

  if (is.null(keys) || length(keys) == 0L) {
    cli::cli_abort("No {.field state_dict_keys} supplied; cannot load weights.")
  }

  missing_in_weights <- setdiff(keys, names(weights))
  if (length(missing_in_weights) > 0L) {
    cli::cli_abort(c(
      "{length(missing_in_weights)} checkpoint key{?s} missing from the weights file.",
      x = "First few: {.val {head(missing_in_weights, 3)}}"
    ))
  }

  consumed_params  <- character()
  consumed_buffers <- character()
  unmatched        <- character()
  shape_mismatch   <- character()
  dtype_mismatch   <- character()

  for (sd_key in keys) {
    r_path <- translate_fn(sd_key)
    tensor <- weights[[sd_key]]

    target <- NULL
    kind   <- NA_character_
    if (r_path %in% names(all_params)) {
      target <- all_params[[r_path]]
      kind   <- "param"
    } else if (r_path %in% names(all_buffers)) {
      target <- all_buffers[[r_path]]
      kind   <- "buffer"
    }

    if (is.null(target)) {
      unmatched <- c(unmatched, sd_key)
      next
    }

    if (!identical(as.integer(target$size()), as.integer(tensor$size()))) {
      shape_mismatch <- c(
        shape_mismatch,
        sprintf("%s  (R %s  vs  ckpt %s)",
                sd_key,
                paste(as.integer(target$size()), collapse = "x"),
                paste(as.integer(tensor$size()), collapse = "x"))
      )
      next
    }

    # `set_data()` takes whatever dtype it is handed, so a bf16 or fp16
    # checkpoint would land in float32 slots and run at the wrong
    # precision with nothing said. Cast, and say so once.
    # By name: two dtype objects for the same dtype are different R
    # objects, so `identical()` on them is always FALSE and would report
    # every tensor in the checkpoint as a mismatch.
    if (!identical(as.character(tensor$dtype), as.character(target$dtype))) {
      dtype_mismatch <- c(dtype_mismatch,
                          sprintf("%s (%s -> %s)", sd_key,
                                  as.character(tensor$dtype),
                                  as.character(target$dtype)))
      tensor <- tensor$to(dtype = target$dtype)
    }

    if (kind == "param") {
      target$set_data(tensor)
      consumed_params <- c(consumed_params, r_path)
    } else {
      target$copy_(tensor)
      consumed_buffers <- c(consumed_buffers, r_path)
    }
  }

  if (length(unmatched) > 0L || length(shape_mismatch) > 0L) {
    msgs <- c()
    if (length(unmatched) > 0L) {
      msgs <- c(
        msgs,
        x = "{length(unmatched)} checkpoint key{?s} had no matching R module:",
        set_names(head(unmatched, 5), rep("*", min(5, length(unmatched))))
      )
    }
    if (length(shape_mismatch) > 0L) {
      msgs <- c(
        msgs,
        x = "{length(shape_mismatch)} shape mismatch{?es}:",
        set_names(head(shape_mismatch, 5),
                  rep("*", min(5, length(shape_mismatch))))
      )
    }
    cli::cli_abort(c("Weight loading failed", msgs))
  }

  if (length(dtype_mismatch) > 0L) {
    cli::cli_warn(c(
      "{length(dtype_mismatch)} checkpoint tensor{?s} {?was/were} cast to \\
       the module's dtype.",
      set_names(utils::head(dtype_mismatch, 3),
                rep("*", min(3, length(dtype_mismatch)))),
      i = "This package runs float32 end to end. A half- or bfloat16 \\
           checkpoint is upcast, which costs memory rather than accuracy \\
           -- but it is not what the publisher measured."
    ))
  }

  unfilled_params  <- setdiff(names(all_params),  consumed_params)
  unfilled_buffers <- setdiff(names(all_buffers), consumed_buffers)
  if (length(unfilled_params) > 0L || length(unfilled_buffers) > 0L) {
    msgs <- c(
      x = "{length(unfilled_params)} parameter{?s} and \\
           {length(unfilled_buffers)} buffer{?s} were not populated from the checkpoint.",
      set_names(head(c(unfilled_params, unfilled_buffers), 8),
                rep("*", min(8, length(c(unfilled_params, unfilled_buffers)))))
    )
    if (isTRUE(strict)) {
      cli::cli_abort(c("Weight loading incomplete", msgs))
    }
    cli::cli_warn(c("Weight loading incomplete", msgs))
  } else {
    cli::cli_alert_success(
      "Loaded {length(consumed_params)} parameter{?s}\\
      {if (length(consumed_buffers)) paste0(' + ', length(consumed_buffers), ' buffer(s)') else ''}."
    )
  }

  invisible(model)
}

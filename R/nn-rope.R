# Rotary position embedding.

#' Rotary position embedding with a checkpoint-loaded frequency table
#'
#' Rotates pairs of channels of `x` by an angle proportional to position
#' along the sequence axis.
#'
#' The inverse frequencies are **loaded from the checkpoint**, not
#' recomputed from `base`. TabFM's were computed in bfloat16 at training
#' time; regenerating them in float32 gives values that differ by ~1e-3,
#' and the error compounds with sequence length. TabICL's are a learnable
#' parameter and so are not a closed form at all. `base` only seeds the
#' table before loading.
#'
#' Two pairing conventions are in use, and they are not interchangeable:
#'
#' * `interleaved = TRUE` pairs adjacent channels `(0,1), (2,3), ...`
#'   (TabFM, LLaMA).
#' * `interleaved = FALSE` splits the head dimension in half and pairs
#'   `(i, i + d/2)` (TabICL, whose `row_rope_interleaved` config field is
#'   `False`).
#'
#' @param dim Head dimension. The table holds `dim / 2` frequencies.
#' @param base Frequency base used for the initial (overwritten) values.
#' @param interleaved Pairing convention, see above.
#' @param learnable When `TRUE` the table is a parameter rather than a
#'   buffer, matching a checkpoint that trained it.
#' @param seq_axis Which axis carries position. TabFM rotates
#'   `(B, T, N, Dh)` along axis 2; TabICL rotates `(..., H, T, Dh)` along
#'   the second-to-last axis, which is what `NULL` selects.
#' @keywords internal
rope <- torch::nn_module(
  "RoPE",

  initialize = function(dim, base = 100000.0, interleaved = TRUE,
                        learnable = FALSE, seq_axis = NULL) {
    # torch_arange(0, dim - 2, 2) gives 0, 2, ..., dim-2 -- the even
    # channel indices, matching `torch.arange(0, dim, 2)`.
    exps <- torch::torch_arange(0L, dim - 2L, 2L,
                                dtype = torch::torch_float())$div(dim)
    freqs <- 1 / base^exps
    if (isTRUE(learnable)) {
      self$freqs <- torch::nn_parameter(freqs)
    } else {
      self$register_buffer("freqs", freqs)
    }
    self$interleaved <- isTRUE(interleaved)
    self$seq_axis <- seq_axis
    # One entry, keyed on the sequence length: consecutive calls into a
    # stack rotate the same `T`, and rebuilding the `(T, Dh)` cos/sin
    # pair per attention per layer per member per chunk is the same
    # arithmetic every time. Reference LLaMA implementations cache the
    # same way.
    #
    # It caches a *learnable* table too, which is only sound because this
    # package does not train: the network is built, loaded, put in
    # `eval()` and never updated, so `freqs` is frozen for the object's
    # lifetime. A fine-tuning path would have to drop this cache -- and
    # would have a great deal else to change first.
    self$.angle_cache <- new.env(parent = emptyenv())
  },

  # The channel-pair index vectors depend on the head dimension alone,
  # which cannot change between calls, so they are built on first use and
  # kept.
  .pair_indices = function(dh, dev) {
    key <- paste0("idx-", dh, "-", as.character(dev))
    hit <- self$.angle_cache[[key]]
    if (!is.null(hit)) return(hit)
    idx <- if (self$interleaved) {
      list(torch::torch_arange(1L, dh - 1L, 2L, dtype = torch::torch_long(),
                               device = dev),
           torch::torch_arange(2L, dh, 2L, dtype = torch::torch_long(),
                               device = dev))
    } else {
      half <- as.integer(dh %/% 2L)
      list(torch::torch_arange(1L, half, dtype = torch::torch_long(),
                               device = dev),
           torch::torch_arange(half + 1L, dh, dtype = torch::torch_long(),
                               device = dev))
    }
    assign(key, idx, envir = self$.angle_cache)
    idx
  },

  # The (T, Dh) cos/sin tables, for this sequence length on this device.
  .angles = function(t, dev) {
    key <- paste0("ang-", t, "-", as.character(dev))
    hit <- self$.angle_cache[[key]]
    if (!is.null(hit)) return(hit)
    pos <- torch::torch_arange(0L, t - 1L, dtype = torch::torch_float(),
                               device = dev)
    f <- torch::torch_outer(pos, self$freqs$to(dtype = torch::torch_float()))
    ang <- if (self$interleaved) {
      # Each frequency is duplicated into its adjacent pair, so channels
      # (2k, 2k+1) share an angle.
      list(cos = torch::torch_repeat_interleave(f$cos(), 2L, dim = -1L),
           sin = torch::torch_repeat_interleave(f$sin(), 2L, dim = -1L))
    } else {
      # The frequency table is concatenated with itself, so channel i and
      # channel i + d/2 share an angle.
      list(cos = torch::torch_cat(list(f$cos(), f$cos()), dim = -1L),
           sin = torch::torch_cat(list(f$sin(), f$sin()), dim = -1L))
    }
    # One length at a time: a stack that grows T would otherwise keep
    # every table it ever saw.
    rm(list = grep("^ang-", ls(self$.angle_cache), value = TRUE),
       envir = self$.angle_cache)
    assign(key, ang, envir = self$.angle_cache)
    ang
  },

  forward = function(x) {
    nd <- x$dim()
    axis <- self$seq_axis %||% (nd - 1L)
    t  <- x$size(axis)
    dh <- x$size(nd)
    dev <- x$device

    ang <- self$.angles(t, dev)
    cos <- ang$cos; sin <- ang$sin

    idx <- self$.pair_indices(dh, dev)
    x1 <- torch::torch_index_select(x, dim = -1L, index = idx[[1]])
    x2 <- torch::torch_index_select(x, dim = -1L, index = idx[[2]])
    rot <- if (self$interleaved) {
      torch::torch_stack(list(-x2, x1), dim = -1L)$reshape(x$size())
    } else {
      torch::torch_cat(list(-x2, x1), dim = -1L)
    }

    cos <- cos$to(dtype = x$dtype)
    sin <- sin$to(dtype = x$dtype)
    # The angle table is (T, Dh). It broadcasts straight against a
    # sequence axis that is second-to-last; anywhere else it needs axes
    # inserted so T lines up.
    if (!is.null(self$seq_axis) && self$seq_axis != nd - 1L) {
      for (i in seq_len(nd - 1L - self$seq_axis)) {
        cos <- cos$unsqueeze(2L)
        sin <- sin$unsqueeze(2L)
      }
      for (i in seq_len(self$seq_axis - 1L)) {
        cos <- cos$unsqueeze(1L)
        sin <- sin$unsqueeze(1L)
      }
    }
    x * cos + rot * sin
  }
)

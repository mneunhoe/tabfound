# Cut-down architecture configs, shared by `test-arch.R` and
# `test-arch-styles.R`. Building one needs torch; the callers skip
# without it.

# A model object of the shape `tabfound_architecture()` expects, without
# the 600 MB of weights a real one carries.
arch_test_model <- function(config, backend, task = "classification") {
  net <- suppressMessages(get_backend(backend)$build(config, task))
  structure(
    list(spec = NULL, state = NULL, model = net, config = config,
         device = "cpu", backend = backend, task = task,
         model_ref = list(model = "test", backend = backend, device = "cpu",
                          args = list())),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor", "tabfound_model")
  )
}

arch_test_configs <- function() {
  list(
    tabpfn = list(
      config = list(arch = "per_feature_transformer", head = "classifier",
                    n_layers = 2L, embedding_dim = 12L, n_heads = 3L,
                    mlp_hidden_dim = 24L, features_per_group = 3L,
                    n_out_classes = 10L),
      task = "classification"),
    tabpfn26 = list(
      config = list(arch = "tabpfn_v2_6", head = "regressor", n_layers = 2L,
                    embedding_dim = 12L, n_heads = 3L, mlp_hidden_dim = 24L,
                    features_per_group = 3L, num_thinking_rows = 4L,
                    encoder_type = "linear", n_bar_bins = 16L),
      task = "regression"),
    tabpfn3 = list(
      config = list(arch = "tabpfn_v3", head = "classifier", embed_dim = 8L,
                    feat_agg_num_cls_tokens = 2L, ff_factor = 2L,
                    feature_group_size = 3L, use_nan_indicators = TRUE,
                    dist_embed_num_blocks = 1L, dist_embed_num_heads = 2L,
                    dist_embed_num_inducing_points = 4L,
                    feat_agg_num_blocks = 1L, feat_agg_num_heads = 2L,
                    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
                    decoder_head_dim = 4L, decoder_num_heads = 2L,
                    decoder_use_softmax_scaling = TRUE,
                    softmax_scaling_mlp_hidden_dim = 4L, max_num_classes = 6L,
                    n_out = 6L, n_bar_bins = 16L),
      task = "classification"),
    # v3.5 carries no `head`: one checkpoint serves both tasks, so `task`
    # is what picks which decoder the diagram draws. Run as a regressor
    # here so the pair covers both branches of `tabpfn35_describe()`.
    tabpfn35 = list(
      config = list(arch = "tabpfn_v3_5", head = "multitask", embed_dim = 8L,
                    feat_agg_num_cls_tokens = 2L, ff_factor = 2L,
                    feature_group_size = 3L, use_nan_indicators = TRUE,
                    fourier_encoding_num_frequencies = 4L,
                    cell_ecdf_num_frequencies = 2L, cell_ecdf_num_buckets = 64L,
                    dist_embed_num_blocks = 1L, dist_embed_num_heads = 2L,
                    dist_embed_num_inducing_points = 4L,
                    feat_agg_num_blocks = 1L, feat_agg_num_heads = 2L,
                    nlayers = 2L, icl_num_heads = 2L, icl_num_kv_heads_test = 1L,
                    decoder_head_dim = 4L, decoder_num_heads = 2L,
                    decoder_use_softmax_scaling = TRUE,
                    softmax_scaling_mlp_hidden_dim = 4L, max_num_classes = 6L,
                    num_buckets = 16L, n_bar_bins = 16L),
      task = "regression"),
    tabicl = list(
      config = list(arch = "tabicl", head = "classifier", max_classes = 6L,
                    num_quantiles = 9L, embed_dim = 8L, col_num_blocks = 1L,
                    col_nhead = 2L, col_num_inds = 4L, col_feature_group = "same",
                    col_feature_group_size = 3L, col_target_aware = TRUE,
                    col_ssmax = "qassmax-mlp-elementwise", row_num_blocks = 2L,
                    row_nhead = 2L, row_num_cls = 2L, row_rope_base = 100000,
                    row_rope_interleaved = FALSE, icl_num_blocks = 2L,
                    icl_nhead = 2L, icl_ssmax = "qassmax-mlp-elementwise",
                    ff_factor = 2L, activation = "gelu", bias_free_ln = FALSE),
      task = "classification"),
    tabfm = list(
      config = list(arch = "tabfm", embed_dim = 8L, ff_factor = 2L,
                    row_num_cls = 2L, col_num_blocks = 1L, col_nhead = 2L,
                    col_num_inds = 4L, row_num_blocks = 1L, row_nhead = 2L,
                    icl_num_blocks = 2L, icl_nhead = 2L, max_classes = 6L,
                    feature_group_size = 3L, num_freq = 4L,
                    is_classifier = TRUE, decoder_hidden = NULL),
      task = "classification"),
    # Medium's shape (a row projection, grouped-query attention) so the
    # optional `row_project` stage is the one exercised.
    kumo = list(
      config = list(arch = "kumo_tabular", head = "classifier", size = "medium",
                    cell_channels = 8L, num_embedding_layers = 2L,
                    num_embedding_heads = 2L, num_inducing_points = 4L,
                    group_size = 3L, num_frequencies = 4L,
                    num_readout_tokens = 2L, icl_channels = 8L,
                    num_icl_layers = 2L, num_icl_heads = 2L,
                    num_icl_key_value_heads_for_query = 1L, gate_hidden = 4L,
                    num_classes = 10L, num_quantiles = 0L),
      task = "classification"),
    mitra = list(
      config = list(dim = 8L, dim_output = 6L, n_heads = 2L, n_layers = 2L,
                    task = "CLASSIFICATION"),
      task = "classification")
  )
}

# Every config as both a classifier and a regressor, plus Kumo small,
# which has no row projection. The key and vocabulary tests run on all
# of them: a regressor draws different decoders and outputs, and so
# different symbols and abbreviations.
arch_test_variants <- function() {
  cs <- arch_test_configs()
  flip <- list(
    tabpfn   = list(head = "regressor", n_bar_bins = 16L),
    tabpfn26 = list(head = "classifier", n_out_classes = 6L),
    tabpfn3  = list(head = "regressor", n_out = 16L, num_buckets = 16L),
    tabpfn35 = list(),
    tabicl   = list(head = "regressor", max_classes = 0L),
    tabfm    = list(is_classifier = FALSE),
    kumo     = list(head = "regressor", num_classes = 0L, num_quantiles = 9L),
    mitra    = list(task = "REGRESSION", dim_output = 1L)
  )
  other <- c(classification = "regression", regression = "classification")
  out <- list()
  for (nm in names(cs)) {
    out[[paste(nm, cs[[nm]]$task)]] <- list(
      backend = nm, config = cs[[nm]]$config, task = cs[[nm]]$task)
    task2 <- other[[cs[[nm]]$task]]
    out[[paste(nm, task2)]] <- list(
      backend = nm, config = utils::modifyList(cs[[nm]]$config, flip[[nm]]),
      task = task2)
  }
  # Small: the readout tokens flatten straight to the in-context width.
  out[["kumo small"]] <- list(
    backend = "kumo", task = "classification",
    config = utils::modifyList(cs$kumo$config,
                               list(size = "small", icl_channels = 16L,
                                    num_icl_key_value_heads_for_query = NULL)))
  out
}

arch_test_variant_arch <- function(v) {
  tabfound_architecture(arch_test_model(v$config, v$backend, v$task))
}

# --- Tracing the forward pass ---------------------------------------------
#
# R torch has no module forward hooks, so the order test wraps the
# reference a parent module holds to each submodule it wants to watch.
# The wrapper forces its arguments before logging: R evaluates arguments
# lazily, so without that `self$b(self$a(x))` would log b before a.

# The module at a dotted path, or NULL when the path names a parameter
# or nothing at all.
arch_module_at <- function(net, path) {
  m <- net
  for (seg in strsplit(path, ".", fixed = TRUE)[[1L]]) {
    kids <- attr(m, "module")$children
    if (!seg %in% names(kids)) return(NULL)
    m <- kids[[seg]]
  }
  m
}

# Every module path in a network, parents before children.
arch_module_paths <- function(net, path = "") {
  kids <- attr(net, "module")$children
  out <- character()
  for (nm in names(kids)) {
    p <- if (nzchar(path)) paste0(path, ".", nm) else nm
    out <- c(out, p, arch_module_paths(kids[[nm]], p))
  }
  out
}

# Run `forward()` and give the watched `paths` in the order they were
# first called. Every module in the network is wrapped to force its
# arguments before it runs, which is the order Python's eager evaluation
# gives: without it, R's lazy evaluation runs `f(x, g(y))` by entering f
# first and only computing g(y) when f's body first reads it. A call to
# watched module places it; a module never called directly is placed by
# its first call to a module inside it -- so an attention step whose
# projections are reached into, rather than called, is still seen. (A
# direct call wins because an attention may run its key/value projection
# early, before the norm on its queries.)
arch_trace_first_calls <- function(net, paths, forward) {
  log <- character()
  direct <- logical()
  watched <- paths[!vapply(paths, function(p) is.null(arch_module_at(net, p)),
                           logical(1))]
  all <- arch_module_paths(net)
  # The watched path a module belongs to: itself, or its nearest watched
  # ancestor.
  owner <- vapply(all, function(m) {
    hit <- watched[m == watched | startsWith(m, paste0(watched, "."))]
    if (length(hit)) hit[which.max(nchar(hit))] else NA_character_
  }, character(1))
  # Deepest first, so a parent's wrapper never hides a child's.
  depth <- lengths(strsplit(all, ".", fixed = TRUE))
  for (i in order(-depth)) local({
    path <- all[i]
    tag <- owner[[i]]   # each wrapper keeps its own, not the loop's last
    segs <- strsplit(path, ".", fixed = TRUE)[[1L]]
    parent <- if (length(segs) > 1L)
      arch_module_at(net, paste(segs[-length(segs)], collapse = ".")) else net
    nm <- segs[length(segs)]
    old <- parent[[nm]]
    f <- function(...) {
      args <- list(...)
      if (!is.na(tag)) {
        log <<- c(log, tag)
        direct <<- c(direct, identical(tag, path))
      }
      do.call(old, args)
    }
    attributes(f) <- attributes(old)
    parent[[nm]] <- f
  })
  torch::with_no_grad(forward())
  at <- vapply(unique(log), function(w) {
    hit <- which(log == w & direct)
    if (length(hit)) hit[1] else which(log == w)[1]
  }, integer(1))
  names(at)[order(at)]
}

# One small forward pass through a test variant's network.
arch_test_forward <- function(v, net) {
  torch::torch_manual_seed(3L)
  clf <- identical(v$task, "classification")
  n_tr <- 12L; n_te <- 5L; p <- 5L
  y_of <- function(n) {
    if (clf) torch::torch_randint(0L, 3L, c(1L, n))$to(dtype = torch::torch_float())
    else torch::torch_randn(c(1L, n))
  }
  x <- torch::torch_randn(c(1L, n_tr + n_te, p))
  x_tr <- x[, 1:n_tr, ]
  x_te <- x[, (n_tr + 1L):(n_tr + n_te), ]
  switch(v$backend,
    tabpfn = , tabpfn3 = , tabpfn35 = function() net(x_tr, y_of(n_tr), x_te),
    # v2.6 ships its column embeddings rather than drawing them.
    tabpfn26 = function() net(x_tr, y_of(n_tr), x_te,
                              column_embeddings = torch::torch_randn(c(64L, net$pos_emb_dim))),
    tabicl = function() net(x, y_of(n_tr)),
    tabfm = function() {
      y <- torch::torch_full(c(1L, n_tr + n_te), -100)
      y[1, 1:n_tr] <- y_of(n_tr)[1, ]
      net(x, y, torch::torch_tensor(n_tr, dtype = torch::torch_long())$reshape(1L))
    },
    mitra = function() net(x_tr, y_of(n_tr), x_te),
    kumo = function() {
      y <- if (clf) torch::torch_randint(0L, 3L, c(1L, n_tr), dtype = torch::torch_long())
           else torch::torch_randn(c(1L, n_tr))
      net(x, y, c(TRUE, rep(FALSE, p - 1L)))
    }
  )
}

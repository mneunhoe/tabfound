# A stage's children are the steps of its forward pass, in the order
# they run. That is hand-written in each describe hook, so it is checked
# here against the network itself: every child's module is wrapped, one
# small forward pass runs, and the order the modules are first called in
# must be the order the children are listed in.

skip_if_not_installed("torch")

# What the trace saw for one variant: a row per child with its role and
# the position its module was first called at (NA if never called, or
# if its prefix names a parameter rather than a module).
arch_trace_children <- function(v) {
  m <- arch_test_model(v$config, v$backend, v$task)
  net <- m$model
  net$eval()
  arch <- tabfound_architecture(m)
  rows <- do.call(rbind, lapply(arch$stages, function(s) {
    if (!length(s$children)) return(NULL)
    data.frame(
      stage = s$id,
      child = vapply(s$children, `[[`, character(1), "id"),
      role = vapply(s$children, `[[`, character(1), "role"),
      feeds = vapply(s$children, function(ch) ch$feeds %||% NA_character_,
                     character(1)),
      path = vapply(s$children, function(ch) ch$prefix[1] %||% NA_character_,
                    character(1)),
      stringsAsFactors = FALSE
    )
  }))
  rows$module <- !is.na(rows$path) &
    !vapply(rows$path, function(p) is.na(p) || is.null(arch_module_at(net, p)),
            logical(1))
  order <- arch_trace_first_calls(net, rows$path[rows$module],
                                  arch_test_forward(v, net))
  rows$called <- match(rows$path, order)
  list(arch = arch, rows = rows)
}

test_that("children are listed in the order the forward pass runs them", {
  for (nm in names(arch_test_variants())) {
    tr <- arch_trace_children(arch_test_variants()[[nm]])
    for (st in unique(tr$rows$stage)) {
      r <- tr$rows[tr$rows$stage == st, ]
      info <- paste(nm, st)
      steps <- r[r$role == "step", ]
      # Every step is a module the trace can watch, and it runs.
      expect_true(all(steps$module), info = paste(info, "unwatchable:",
        paste(steps$child[!steps$module], collapse = ", ")))
      expect_false(anyNA(steps$called), info = paste(info, "never ran:",
        paste(steps$child[is.na(steps$called)], collapse = ", ")))
      expect_true(all(diff(steps$called) > 0, na.rm = TRUE),
                  info = paste(info, "ran as:",
                               paste(steps$child[order(steps$called)],
                                     collapse = " ")))
      # A step that runs once after the stack comes after every step.
      fin <- r[r$role == "final", ]
      if (nrow(fin) && nrow(steps)) {
        expect_true(all(fin$called > max(steps$called)), info = info)
      }
      # An input feeds a step of the same stage, and runs if it is a
      # module (a parameter table is read, not called).
      inp <- r[r$role == "input", ]
      expect_true(all(inp$feeds %in% steps$child), info = info)
      expect_false(anyNA(inp$called[inp$module]), info = info)
    }
  }
})

test_that("a stage's children account for every parameter it owns", {
  for (nm in names(arch_test_variants())) {
    v <- arch_test_variants()[[nm]]
    arch <- arch_test_variant_arch(v)
    for (s in arch$stages) {
      if (!length(s$children)) next
      claimed <- sum(vapply(s$children, function(ch)
        if (is.na(ch$params_all)) 0 else ch$params_all, numeric(1)))
      expect_equal(claimed, s$params, info = paste(nm, s$id))
    }
  }
})


# ---------------------------------------------------------------------------
# Residual connections, joins, attention variants and fact anchors
# ---------------------------------------------------------------------------

# A stage's block as a short string: one letter per step child by kind
# (a attention, f feed-forward, n norm, e embedding), with each residual
# connection's steps in brackets. Inputs are left out; final steps follow.
residual_signature <- function(s) {
  letter <- c(attention = "a", ffn = "f", norm = "n", embed = "e",
              decode = "d", input = "i", output = "o")
  steps <- Filter(function(ch) !identical(ch$role, "input"), s$children)
  out <- ""
  open <- NA_integer_
  for (ch in steps) {
    if (!identical(ch$residual, open)) {
      if (!is.na(open)) out <- paste0(out, ")")
      if (!is.na(ch$residual)) out <- paste0(out, "(")
      open <- ch$residual
    }
    out <- paste0(out, letter[[ch$kind]])
  }
  if (!is.na(open)) out <- paste0(out, ")")
  out
}

# The block shapes the forward code has, per backend and stage, read off
# each block's forward(): post-norm TabPFN v2.x, sandwich TabFM, pre-norm
# everywhere else; Kumo norms keys/values and queries apart; final steps
# after the stack.
ARCH_EXPECTED_BLOCKS <- list(
  tabpfn   = c(layers = "(a)n(a)n(f)n"),
  tabpfn26 = c(blocks = "(a)n(a)n(f)n"),
  tabpfn3  = c(dist = "(nna)(nf)(nna)(nf)", agg = "(na)(nf)n", icl = "(na)(nf)"),
  tabpfn35 = c(x_embed = "een", dist = "(nna)(nf)(nna)(nf)",
               agg = "(na)(nf)n", icl = "(na)(nf)"),
  tabicl   = c(col_tf = "(na)(nf)(na)(nf)", row_tf = "(na)(nf)n",
               icl = "(na)(nf)"),
  tabfm    = c(col1 = "(nan)(nfn)(nan)(nfn)en", row1 = "(nan)(nfn)n",
               col2 = "(nan)(nfn)(nan)(nfn)en", row2 = "(nan)(nfn)n",
               icl = "(nan)(nfn)"),
  mitra    = c(layers = "(na)(nf)(na)(nf)"),
  kumo     = c(col = "(nna)(nf)(nna)(nf)", row = "(nna)(nf)n",
               icl = "(nna)(nf)n")
)

test_that("each block's residual connections are as its forward code adds them", {
  for (nm in names(arch_test_variants())) {
    v <- arch_test_variants()[[nm]]
    arch <- arch_test_variant_arch(v)
    with_kids <- Filter(function(s) length(s$children) > 0L, arch$stages)
    got <- vapply(with_kids, residual_signature, character(1))
    names(got) <- vapply(with_kids, `[[`, character(1), "id")
    expect_identical(got, ARCH_EXPECTED_BLOCKS[[v$backend]], info = nm)
    # Residual groups are numbered in order and never interleave.
    for (s in with_kids) {
      r <- vapply(s$children, `[[`, integer(1), "residual")
      r <- r[!is.na(r)]
      expect_identical(unique(r), seq_along(unique(r)), info = paste(nm, s$id))
      expect_false(is.unsorted(r), info = paste(nm, s$id))
    }
  }
})

test_that("every attention step says where its keys come from", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    for (s in arch$stages) {
      for (ch in s$children) {
        if (!identical(ch$kind, "attention")) next
        expect_s3_class(ch$attn, "arch_attn")
      }
    }
  }
  expect_error(arch_attn("everything"))
  expect_error(arch_attn(queries = "all of them"))
  expect_error(arch_stage("x", "x", kind = "ffn", attn = arch_attn()), "attention child")
  expect_error(arch_stage("x", "x", join_rows = "training rows"), "needs")
})

test_that("target encoders and prepended tokens say how they join the stream", {
  joins <- c(y_encoder = "concat", y_embed = "concat", col_y = "add",
             icl_y = "add", row_y = "add", cls = "prepend",
             readout = "prepend", think = "prepend", pos = "add")
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    for (s in arch$stages) {
      if (!s$id %in% names(joins)) next
      want <- joins[[s$id]]
      # Mitra's target embedding is put in front, as feature column 1.
      if (identical(arch$backend, "mitra") && identical(s$id, "y_embed")) want <- "prepend"
      expect_identical(s$join, want, info = paste(nm, s$id))
    }
  }
})

test_that("every anchored fact finds its place in every model", {
  for (nm in names(arch_test_variants())) {
    arch <- arch_test_variant_arch(arch_test_variants()[[nm]])
    ids <- vapply(arch$stages, `[[`, character(1), "id")
    for (f in intersect(names(arch$facts), names(ARCH_FACT_ANCHORS))) {
      a <- arch_fact_anchor(arch, f)
      expect_false(is.null(a), info = paste(nm, f))
      s <- arch$stages[[match(a$stage, ids)]]
      if (!is.null(a$child)) {
        expect_true(a$child %in% vapply(s$children, `[[`, character(1), "id"),
                    info = paste(nm, f))
      }
    }
  }
  expect_null(arch_fact_anchor(list(stages = list()), "Layers"))
  expect_null(arch_fact_anchor(list(stages = list()), "KV cache"))
})

test_that("residual groups flatten into numbered children", {
  st <- arch_stage("s", "s", children = list(
    arch_stage("a", "a", kind = "attention"),
    arch_residual(arch_stage("n", "n", kind = "norm"),
                  arch_stage("f", "f", kind = "ffn")),
    arch_stage("m", "m", kind = "norm")
  ))
  expect_identical(vapply(st$children, `[[`, character(1), "id"),
                   c("a", "n", "f", "m"))
  expect_identical(vapply(st$children, `[[`, integer(1), "residual"),
                   c(NA, 1L, 1L, NA))
})

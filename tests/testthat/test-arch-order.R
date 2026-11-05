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

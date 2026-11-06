# testthat runs every test under the C collation, which is the one locale
# where R's `sort()` and sklearn's code-point order happen to agree. A test
# of locale independence has to switch to one where they do not.
local_natural_collate <- function(env = parent.frame()) {
  old <- Sys.getlocale("LC_COLLATE")
  set <- suppressWarnings(Sys.setlocale("LC_COLLATE", "en_US.UTF-8"))
  if (!nzchar(set)) testthat::skip("en_US.UTF-8 collation is not available")
  withr::defer(Sys.setlocale("LC_COLLATE", old), envir = env)
  # Some platforms accept the name but still collate bytewise, and then the
  # test would pass without checking anything.
  if (identical(sort(c("a", "B")), c("B", "a"))) {
    testthat::skip("en_US.UTF-8 collates like the C locale here")
  }
  invisible(set)
}

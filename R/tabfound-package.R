#' tabfound: Tabular Foundation Models in Pure R Torch
#'
#' Inference-only R implementations of open-weight tabular foundation
#' models, built on the R `torch` package. No Python runtime is needed
#' at predict time.
#'
#' The package is organised around a *backend registry*. A backend
#' describes one model family: how to build its network from a config,
#' how to map checkpoint keys onto R module paths, how to preprocess a
#' design matrix, and how to turn network outputs into predictions.
#' See [register_backend()] and [list_backends()].
#'
#' User-facing entry points are [tabular_classifier()] and
#' [tabular_regressor()].
#'
#' @keywords internal
#' @importFrom stats approx median optimize predict quantile sd
#' @importFrom utils head tail
"_PACKAGE"

# NULL-coalescing operator used across the package.
`%||%` <- function(x, y) if (is.null(x)) y else x

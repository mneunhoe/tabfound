# Multiple-imputation simulation harness.
#
#   Rscript inst/simulation/run-mi-sim.R [--backend=lm] [--reps=200]
#                                        [--n=400] [--m=5] [--maxit=3]
#                                        [--proper=TRUE] [--seed=20260807]
#                                        [--save]
#
# The MI code has no Python reference to be checked against, so this is
# what stands in for the parity harness: a simulation where the right
# answer is known by construction.
#
# The design is the textbook MAR demonstration. Missingness depends on
# `y`, which is fully observed -- so it is missing *at random* given the
# data, but it is not ignorable for a complete-case analysis, because
# dropping rows selects on the outcome. That gives three things a run can
# check at once:
#
#   * FULL   (before deletion)     the benchmark. Unbiased by definition.
#   * CC     (complete cases)      must be visibly biased. If it is not,
#                                  the mechanism is too weak to test anything.
#   * MI     (tabfound)            should recover FULL's estimate, with
#                                  wider intervals and ~95% coverage.
#
# A `mice` PMM arm runs alongside as an external yardstick: not a target
# to beat, just the answer a well-understood imputer gives on the same
# data.
#
# `--backend=lm` uses a correctly specified linear imputation model built
# out of base R -- no weights, no torch, seconds to run. That arm tests
# the *loop*: if the chained equations condition on the wrong rows, or
# draw means instead of samples, the lm arm breaks too. Only once it is
# clean does a foundation-model arm say anything about the model.
#
# Backends are located the same way the parity harness does it:
#
#   TABFOUND_TABPFN_CLF_DIR / TABFOUND_TABPFN_REG_DIR
#   TABFOUND_TABICL_CLF_DIR / TABFOUND_TABICL_REG_DIR

suppressMessages({
  pkgload::load_all(".", quiet = TRUE)
})

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}
BACKEND <- opt("backend", "lm")
REPS    <- as.integer(opt("reps", 200L))
N       <- as.integer(opt("n", 400L))
M       <- as.integer(opt("m", 5L))
MAXIT   <- as.integer(opt("maxit", 3L))
# Coverage is the reason this argument exists, so the harness has to be
# able to run both sides of it. The default matches the package's:
# `--proper=TRUE` bootstraps the context once per imputation,
# `--proper=bayes` uses Dirichlet weights instead. This is the switch the
# properness section of README.md was measured with.
PROPER  <- local({
  v <- opt("proper", "FALSE")
  if (toupper(v) %in% c("TRUE", "FALSE")) as.logical(toupper(v)) else v
})
SEED    <- as.integer(opt("seed", 20260807L))
SAVE    <- "--save" %in% args
WITH_MICE <- requireNamespace("mice", quietly = TRUE)


# ---------------------------------------------------------------------------
# The data-generating process
# ---------------------------------------------------------------------------

TRUTH <- c(x1 = 2, x2 = -1, gb = 1.5)

# `x1` and `x2` are correlated, so an imputation model that ignores the
# other covariates is not merely inefficient, it is wrong -- which is
# what makes the chained part of chained equations do any work.
gen_data <- function(n) {
  x2 <- stats::rnorm(n)
  x1 <- 0.5 * x2 + sqrt(0.75) * stats::rnorm(n)
  g  <- factor(sample(c("a", "b"), n, replace = TRUE, prob = c(0.6, 0.4)))
  y  <- 1 + TRUTH[["x1"]] * x1 + TRUTH[["x2"]] * x2 +
        TRUTH[["gb"]] * (g == "b") + stats::rnorm(n)
  data.frame(y = y, x1 = x1, x2 = x2, g = g)
}

# MAR: the missingness of `x1` and of `g` both depend on `y`, and on
# nothing unobserved. `y` is in the imputation model, so a correct
# imputer conditions the missingness away; a complete-case analysis
# cannot, because it has selected on the outcome.
amputate <- function(d) {
  z <- as.numeric(scale(d$y))
  d$x1[stats::runif(nrow(d)) < stats::plogis(-0.4 + 0.9 * z)] <- NA
  d$g[stats::runif(nrow(d)) < stats::plogis(-1.2 + 0.8 * z)]  <- NA
  d
}

ANALYSIS <- y ~ x1 + x2 + g
COEFS    <- c(x1 = "x1", x2 = "x2", gb = "gb")   # gb = the g == "b" contrast


# ---------------------------------------------------------------------------
# Imputation models
# ---------------------------------------------------------------------------

# A correctly specified linear/frequency imputer with the same object
# shape a real backend has. Nothing in the MI code touches the network,
# so this drops straight into `tabfound_models()`.
lm_model <- function(task) {
  spec <- if (task == "classification") {
    list(
      fit = function(X, y) {
        y <- droplevels(as.factor(y))
        X <- cbind(1, as.matrix(X)); storage.mode(X) <- "double"
        list(X = X, y = y, class_levels = levels(y))
      },
      predict = function(state, newdata, type = "class", ...) {
        # Logistic regression, which is the *correct* conditional model
        # for this DGP's binary `g`. An earlier version used a one-vs-rest
        # linear probability model and left a visible residual bias on the
        # `g` coefficient -- misspecification of the imputation model, not
        # a defect in the loop, but it made the arm useless as a
        # gold standard. Multi-class would need a real multinomial fit;
        # this DGP is binary, so refuse rather than approximate.
        Xn <- cbind(1, as.matrix(newdata)); storage.mode(Xn) <- "double"
        lev <- state$class_levels
        if (length(lev) != 2L) {
          stop("the lm arm's classifier only handles two classes", call. = FALSE)
        }
        b <- suppressWarnings(
          stats::glm.fit(state$X, as.numeric(state$y == lev[2]),
                         family = stats::binomial())
        )$coefficients
        b[is.na(b)] <- 0
        p2 <- stats::plogis(as.numeric(Xn %*% b))
        p <- cbind(1 - p2, p2)
        colnames(p) <- as.character(lev)
        if (type == "prob") return(p)
        lev[max.col(p, ties.method = "first")]
      }
    )
  } else {
    list(
      fit = function(X, y) {
        X <- cbind(1, as.matrix(X)); storage.mode(X) <- "double"
        y <- as.numeric(y)
        b <- qr.coef(qr(X), y); b[is.na(b)] <- 0
        n <- nrow(X); p <- sum(b != 0)
        list(b = b, sigma = max(sqrt(sum((y - X %*% b)^2) / max(n - p, 1)), 1e-6))
      },
      predict = function(state, newdata, type = "mean", quantiles = 0.5,
                         n_samples = 1L, seed = NULL, ...) {
        X <- cbind(1, as.matrix(newdata)); storage.mode(X) <- "double"
        mu <- as.numeric(X %*% state$b)
        if (type == "mean") return(mu)
        if (type == "quantiles") {
          out <- outer(mu, stats::qnorm(quantiles) * state$sigma, `+`)
          colnames(out) <- paste0("q", quantiles)
          return(out)
        }
        if (!is.null(seed)) set.seed(seed)
        matrix(stats::rnorm(length(mu) * n_samples, rep(mu, n_samples),
                            state$sigma),
               ncol = as.integer(n_samples))
      },
      types = c("mean", "quantiles", "sample")
    )
  }
  structure(
    list(spec = spec, state = NULL, model = NULL, config = NULL,
         device = "cpu", backend = "lm", task = task,
         model_ref = list(model = "lm", backend = "lm", device = "cpu",
                          args = list())),
    class = c(if (task == "classification") "tabfound_classifier"
              else "tabfound_regressor", "tabfound_model")
  )
}

env_dir <- function(var) {
  v <- Sys.getenv(var, unset = "")
  if (!nzchar(v) || !dir.exists(v)) {
    stop(sprintf("%s is unset or missing; set it to the converted artifacts.", var),
         call. = FALSE)
  }
  v
}

make_models <- function(backend) {
  if (identical(backend, "lm")) {
    return(tabfound_models(classifier = lm_model("classification"),
                           regressor  = lm_model("regression")))
  }
  prefix <- toupper(backend)
  tabfound_models(
    classifier = env_dir(sprintf("TABFOUND_%s_CLF_DIR", prefix)),
    regressor  = env_dir(sprintf("TABFOUND_%s_REG_DIR", prefix)),
    backend    = backend
  )
}


# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------

# Pull (estimate, lower, upper) for the three coefficients out of
# whatever an arm produced.
tidy_lm <- function(f) {
  ci <- stats::confint(f)
  co <- stats::coef(f)
  nm <- c(x1 = "x1", x2 = "x2", gb = "gb")
  data.frame(coef = names(nm), est = co[nm], lo = ci[nm, 1], hi = ci[nm, 2],
             row.names = NULL)
}

tidy_pool <- function(fits) {
  s  <- summary(mice::pool(fits), conf.int = TRUE)
  nm <- c(x1 = "x1", x2 = "x2", gb = "gb")
  i  <- match(nm, s$term)
  data.frame(coef = names(nm), est = s$estimate[i],
             lo = s[i, "2.5 %"], hi = s[i, "97.5 %"], row.names = NULL)
}

one_rep <- function(models, i) {
  # Reseed per replication rather than letting one stream run through the
  # whole simulation. Within a run the arms already share a data set, so
  # they are paired; without this, *across* runs they are not, because
  # each imputer consumes a different amount of randomness before the
  # next replication's data is drawn. Backend A and backend B would then
  # be compared on different data, and at 200 replications the Monte
  # Carlo error of that unpaired difference is large enough to swamp the
  # differences worth seeing. With it, replication i is the same data set
  # for every backend.
  set.seed(SEED + i)
  full <- gen_data(N)
  obs  <- amputate(full)
  out  <- list(
    FULL = tidy_lm(stats::lm(ANALYSIS, data = full)),
    CC   = tidy_lm(stats::lm(ANALYSIS, data = obs))
  )

  imp <- tabfound_impute(obs, m = M, models = models, maxit = MAXIT,
                         proper = PROPER, verbose = FALSE)
  out$MI <- tidy_pool(with(imp, stats::lm(y ~ x1 + x2 + g)))

  if (WITH_MICE) {
    # The yardstick runs last, so it would otherwise inherit whatever RNG
    # state the MI arm happened to leave behind -- different for every
    # backend, which makes PMM wobble across runs for no reason. Reseed so
    # it too is identical everywhere. Everything above this line is
    # upstream of the MI arm and already paired.
    set.seed(SEED + 500000L + i)
    pmm <- mice::mice(obs, m = M, maxit = MAXIT, printFlag = FALSE)
    out$PMM <- tidy_pool(with(pmm, stats::lm(y ~ x1 + x2 + g)))
  }
  out$.nmis <- mean(is.na(obs$x1))
  out
}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cat(sprintf("MAR simulation: backend=%s reps=%d n=%d m=%d maxit=%d proper=%s seed=%d\n",
            BACKEND, REPS, N, M, MAXIT, as.character(PROPER), SEED))
if (!WITH_MICE) cat("  (mice not installed: no PMM arm, no MI pooling)\n")
stopifnot(WITH_MICE)

models <- make_models(BACKEND)
t0 <- Sys.time()
reps <- vector("list", REPS)
for (i in seq_len(REPS)) {
  reps[[i]] <- one_rep(models, i)
  if (i %% max(1L, REPS %/% 20L) == 0L || i == REPS) {
    cat(sprintf("\r  %d/%d  (%.0fs)", i, REPS,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    utils::flush.console()
  }
}
cat("\n")

arms <- setdiff(unique(unlist(lapply(reps, names))), ".nmis")
summarise <- function(arm) {
  rows <- do.call(rbind, lapply(reps, `[[`, arm))
  do.call(rbind, lapply(names(TRUTH), function(k) {
    r <- rows[rows$coef == k, ]
    truth <- TRUTH[[k]]
    data.frame(
      arm      = arm,
      coef     = k,
      truth    = truth,
      estimate = mean(r$est),
      bias     = mean(r$est) - truth,
      emp_sd   = stats::sd(r$est),
      ci_width = mean(r$hi - r$lo),
      coverage = mean(r$lo <= truth & truth <= r$hi)
    )
  }))
}
res <- do.call(rbind, lapply(arms, summarise))
res <- res[order(match(res$coef, names(TRUTH)), match(res$arm, arms)), ]

cat(sprintf("\nmissing in x1: %.1f%%   elapsed: %.0fs\n",
            100 * mean(vapply(reps, `[[`, numeric(1), ".nmis")),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
print(format(res, digits = 3), row.names = FALSE)

cat("\nRead it as: CC should be biased with coverage far below 0.95;\n")
cat("MI should sit near FULL's estimate with coverage near 0.95.\n")

if (SAVE) {
  dir.create("inst/simulation/results", showWarnings = FALSE, recursive = TRUE)
  # The default regime owns the plain name; anything else says so in the
  # filename, so the coverage regimes can sit side by side.
  tag <- if (isFALSE(PROPER)) BACKEND
         else sprintf("%s-%s", BACKEND,
                      if (isTRUE(PROPER)) "proper" else as.character(PROPER))
  f <- sprintf("inst/simulation/results/mi-mar-%s.csv", tag)
  utils::write.csv(res, f, row.names = FALSE)
  cat(sprintf("\nwrote %s\n", f))

  # Per-replication estimates as well as the summary. Aggregates cannot
  # answer whether two arms differ, only that their means are some
  # distance apart -- and since the arms share a data set, a paired test
  # on these is far more powerful than comparing the summary columns.
  long <- do.call(rbind, lapply(seq_along(reps), function(i) {
    r <- reps[[i]]
    do.call(rbind, lapply(setdiff(names(r), ".nmis"), function(a) {
      x <- r[[a]]; x$arm <- a; x$rep <- i; x
    }))
  }))
  f2 <- sprintf("inst/simulation/results/mi-mar-%s-reps.csv", tag)
  utils::write.csv(long, f2, row.names = FALSE)
  cat(sprintf("wrote %s\n", f2))
}

# Paired comparison of two MI arms from one saved run.
#
#   Rscript inst/simulation/paired-compare.R [results/mi-mar-tabpfn-reps.csv]
#                                            [--a=MI] [--b=PMM]
#
# `run-mi-sim.R --save` writes a per-replication file as well as the
# summary, because the arms share a data set replication for replication:
# that pairing is worth far more than comparing summary columns, and it
# is the only way to say whether two arms differ rather than that their
# means are some distance apart.
#
#   coverage   McNemar on the two coverage indicators (discordant pairs
#              only -- the replications where exactly one arm covered)
#   estimate   paired t on the estimates
#   width      paired t on the interval widths
#
# Three coefficients means three tests per family, so a Bonferroni
# threshold of 0.017 is the honest bar for "significant"; the script
# prints the raw p so that judgement stays with the reader.

args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^--", name, "="), "", hit[[1]])
}
file <- {
  pos <- grep("^--", args, invert = TRUE, value = TRUE)
  if (length(pos)) pos[[1]] else "inst/simulation/results/mi-mar-tabpfn-reps.csv"
}
ARM_A <- opt("a", "MI")
ARM_B <- opt("b", "PMM")

long <- utils::read.csv(file, stringsAsFactors = FALSE)
TRUTH <- c(x1 = 2, x2 = -1, gb = 1.5)

pick <- function(arm) {
  d <- long[long$arm == arm, ]
  d[order(d$rep), ]
}
a <- pick(ARM_A)
b <- pick(ARM_B)
stopifnot(nrow(a) == nrow(b), identical(a$rep, b$rep), identical(a$coef, b$coef))

rows <- lapply(names(TRUTH), function(k) {
  truth <- TRUTH[[k]]
  ia <- a[a$coef == k, ]
  ib <- b[b$coef == k, ]
  cov_a <- ia$lo <= truth & truth <= ia$hi
  cov_b <- ib$lo <= truth & truth <= ib$hi
  # McNemar needs both discordant cells; with none, the arms agreed on
  # every replication and there is nothing to test.
  n01 <- sum(!cov_a & cov_b); n10 <- sum(cov_a & !cov_b)
  p_cov <- if (n01 + n10 == 0L) NA_real_
           else stats::mcnemar.test(matrix(c(sum(cov_a & cov_b), n10,
                                             n01, sum(!cov_a & !cov_b)), 2))$p.value
  w_a <- ia$hi - ia$lo; w_b <- ib$hi - ib$lo
  data.frame(
    coef      = k,
    cov_a     = mean(cov_a),
    cov_b     = mean(cov_b),
    p_cov     = p_cov,
    est_diff  = mean(ia$est - ib$est),
    p_est     = stats::t.test(ia$est, ib$est, paired = TRUE)$p.value,
    width_diff = mean(w_a - w_b),
    p_width   = stats::t.test(w_a, w_b, paired = TRUE)$p.value
  )
})
res <- do.call(rbind, rows)
names(res)[2:3] <- paste0("cov_", c(ARM_A, ARM_B))

cat(sprintf("%s vs %s, %d replications, paired on the data set\n",
            ARM_A, ARM_B, length(unique(a$rep))))
print(format(res, digits = 3), row.names = FALSE)
cat("\nBonferroni threshold across three coefficients: 0.017\n")

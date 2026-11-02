#!/usr/bin/env Rscript
# ─────────────────────────────────────────────────────────────────────────────
# Rungs 1 and 2 of the modelling ladder: portfolio sorts, then Fama-MacBeth.
#
# BASE R ONLY. No data.table, no sandwich, no lfe. Three reasons:
#   1. CRAN is blocked from the cloud container, so this is what can be tested here.
#   2. It runs anywhere, including in a closed-book R test on 14 October.
#   3. Newey-West is twelve lines. Writing it out is worth more than importing it,
#      because you can be asked to explain it.
# Swap in data.table for the full 195,000-row panel if speed ever matters; the
# interfaces below are deliberately plain data.frames so that swap is local.
#
# Every implementable-in-real-time rule from PROJECT_PLAN is enforced here:
#   * quintile breakpoints from the CURRENT cross-section only        (Trap #4)
#   * ranks computed within month, never pooled                        (Trap #3)
#   * value weights, and a turnover-based cost haircut on every number
#   * optional industry-neutral sort, which the plan requires
#
# Usage:  Rscript R/sorts_and_fm.R <panel.csv>
#   panel.csv needs: id, month (YYYY-MM-DD), score, mktcap, ret_fwd_1m, industry
# ─────────────────────────────────────────────────────────────────────────────

# ── Newey-West standard error for the mean of a time series ──────────────────
nw_se <- function(x, lags = NULL) {
  x <- x[is.finite(x)]; T <- length(x)
  if (is.null(lags)) lags <- floor(4 * (T / 100)^(2 / 9))   # Newey-West (1994)
  e <- x - mean(x)
  g0 <- sum(e^2) / T
  v <- g0
  if (lags > 0) for (l in 1:lags) {
    gl <- sum(e[(l + 1):T] * e[1:(T - l)]) / T
    v <- v + 2 * (1 - l / (lags + 1)) * gl                  # Bartlett kernel
  }
  sqrt(max(v, 0) / T)
}

tstat_nw <- function(x) mean(x, na.rm = TRUE) / nw_se(x)

# ── rank to [-0.5, +0.5] within a group; the only standardisation allowed ─────
rank_pm <- function(v) {
  ok <- is.finite(v)
  out <- rep(NA_real_, length(v))
  out[ok] <- rank(v[ok], ties.method = "average") / (sum(ok) + 1) - 0.5
  out
}

# ── Rung 1: value-weighted quintile sort ─────────────────────────────────────
portfolio_sort <- function(d, n_bins = 5, cost_bps = 20, industry_neutral = FALSE) {
  d <- d[is.finite(d$score) & is.finite(d$ret_fwd_1m) & is.finite(d$mktcap), ]
  months <- sort(unique(d$month))
  prev_w <- setNames(numeric(0), character(0))
  res <- data.frame()
  for (m in months) {
    g <- d[d$month == m, ]
    if (nrow(g) < 50) next
    # Breakpoints from THIS month only. Never the full sample. Trap #4.
    s <- if (industry_neutral) {
      ave(g$score, g$industry, FUN = rank_pm)     # neutralise within industry
    } else rank_pm(g$score)
    if (all(is.na(s))) next
    q <- cut(rank(s, ties.method = "first"),
             breaks = quantile(rank(s, ties.method = "first"),
                               probs = seq(0, 1, length.out = n_bins + 1)),
             include.lowest = TRUE, labels = FALSE)
    lo <- g[q == 1, ]; hi <- g[q == n_bins, ]
    if (nrow(lo) < 5 || nrow(hi) < 5) next
    wl <- lo$mktcap / sum(lo$mktcap); wh <- hi$mktcap / sum(hi$mktcap)
    r_lo <- sum(wl * lo$ret_fwd_1m); r_hi <- sum(wh * hi$ret_fwd_1m)
    w <- c(setNames(-wl, lo$id), setNames(wh, hi$id))
    ids <- union(names(w), names(prev_w))
    turn <- sum(abs(ifelse(is.na(w[ids]), 0, w[ids]) -
                    ifelse(is.na(prev_w[ids]), 0, prev_w[ids]))) / 2
    prev_w <- w
    res <- rbind(res, data.frame(month = m, lo = r_lo, hi = r_hi,
                                 ls = r_hi - r_lo, turnover = turn,
                                 ls_net = r_hi - r_lo - turn * cost_bps / 1e4))
  }
  res
}

summarise_sort <- function(s, label = "") {
  f <- function(x) c(mean = mean(x) * 100, t = tstat_nw(x),
                     sharpe = mean(x) / sd(x) * sqrt(12))
  g <- f(s$ls); n <- f(s$ls_net)
  cat(sprintf("\n%s  (%d months)\n", label, nrow(s)))
  cat(sprintf("  low quintile      %+.3f %%/mo\n", mean(s$lo) * 100))
  cat(sprintf("  high quintile     %+.3f %%/mo\n", mean(s$hi) * 100))
  cat(sprintf("  long-short GROSS  %+.3f %%/mo   t(NW) = %+.2f   Sharpe %+.2f\n",
              g[1], g[2], g[3]))
  cat(sprintf("  long-short NET    %+.3f %%/mo   t(NW) = %+.2f   Sharpe %+.2f\n",
              n[1], n[2], n[3]))
  cat(sprintf("  turnover          %.2f /month  -> %.3f %%/mo cost drag\n",
              mean(s$turnover), mean(s$turnover) * 20 / 1e4 * 100))
  invisible(c(gross = g[1], t_gross = g[2], net = n[1], t_net = n[2]))
}

# ── Rung 2: Fama-MacBeth ─────────────────────────────────────────────────────
fama_macbeth <- function(d, rhs = c("score"), winsor = 0.01) {
  months <- sort(unique(d$month))
  co <- list()
  for (m in months) {
    g <- d[d$month == m, ]
    g <- g[is.finite(g$ret_fwd_1m), ]
    if (nrow(g) < 50) next
    X <- sapply(rhs, function(v) rank_pm(g[[v]]))      # within-month, Trap #3
    if (is.null(dim(X))) X <- matrix(X, ncol = 1, dimnames = list(NULL, rhs))
    keep <- stats::complete.cases(X)
    if (sum(keep) < 50) next
    y <- g$ret_fwd_1m[keep]
    if (winsor > 0) {
      qq <- quantile(y, c(winsor, 1 - winsor)); y <- pmin(pmax(y, qq[1]), qq[2])
    }
    fit <- lm(y ~ X[keep, , drop = FALSE])
    cf <- coef(fit); names(cf) <- c("(Intercept)", rhs)
    co[[length(co) + 1]] <- cf
  }
  M <- do.call(rbind, co)
  out <- data.frame(term = colnames(M),
                    mean = colMeans(M) * 100,
                    t_nw = apply(M, 2, tstat_nw),
                    n_months = nrow(M), row.names = NULL)
  out
}

print_fm <- function(fm, label = "") {
  cat(sprintf("\n%s  (%d monthly cross-sections)\n", label, fm$n_months[1]))
  for (i in seq_len(nrow(fm)))
    cat(sprintf("  %-14s %+.4f %%/mo   t(NW) = %+.2f\n",
                fm$term[i], fm$mean[i], fm$t_nw[i]))
  invisible(fm)
}

# ── main ─────────────────────────────────────────────────────────────────────
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) stop("usage: Rscript R/sorts_and_fm.R <panel.csv>")
  d <- read.csv(args[1], stringsAsFactors = FALSE)
  stopifnot(all(c("id", "month", "score", "mktcap", "ret_fwd_1m") %in% names(d)))
  if (!"industry" %in% names(d)) d$industry <- "ALL"
  cat(sprintf("panel: %s rows, %d firms, %d months\n",
              format(nrow(d), big.mark = ","), length(unique(d$id)),
              length(unique(d$month))))
  summarise_sort(portfolio_sort(d), "RUNG 1 - quintile sort, unconditional")
  summarise_sort(portfolio_sort(d, industry_neutral = TRUE),
                 "RUNG 1b - quintile sort, industry-neutral")
  print_fm(fama_macbeth(d, "score"), "RUNG 2 - Fama-MacBeth, univariate")
  if ("lnME" %in% names(d))
    print_fm(fama_macbeth(d, c("score", "lnME")), "RUNG 2b - with size control")
}

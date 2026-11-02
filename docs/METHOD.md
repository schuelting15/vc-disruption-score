# Method and specification

## The construct

For a listed firm *i* in month *m* of year *Y*:

1. **Word space.** A TF-IDF vectoriser (sublinear tf, min_df = 5, a stop list of generic and boilerplate words) is fitted on the descriptions of the start-ups in the pool for year *Y*. Those are the start-ups whose first funded year is *Y*−1 or earlier.
2. **Projection.** The firm's latest live 10-K *Item 1: Business* text is projected into that space. Both sides are L2-normalised, so the dot product is the cosine similarity.
3. **Neighbours.** Take the k = 100 most similar start-ups and keep those with similarity ≥ 0.10 (the similarity gate).
4. **Score.** `ds_level_raw` is Σ similarity × funding raised in *Y*−1, over the gated neighbours.

The other pre-registered variants are:

- `ds_change_raw`: the change in that sum against *Y*−2;
- `ds_count`: similarity-weighted round counts;
- `ds_level_scaled`: `ds_level_raw` divided by market cap.

## Pre-registered specification (frozen in `src/full_run.py`)

| Choice | Value |
|---|---|
| Primary score | `ds_level_raw` |
| Primary estimator | Fama-MacBeth on within-month ranks in [−0.5, 0.5], with a size control (log market cap). Next-month return winsorised at 1/99% per month |
| Inference | Newey-West, quoted next to a 12-month circular block bootstrap (2,000 draws), plus a shuffled-score placebo (200 draws) |
| Neighbours / gate | k = 100, cosine ≥ 0.10 |
| Window | 2014-01 to 2024-12 |
| Secondary | Value-weighted quintile sorts with a 20 bp turnover cost. Median-month subperiod split, reported whatever it shows. Fama-French-12 industry-neutral robustness |

Anything added after the full data was scored was labelled exploratory, and its count was tracked.

## Look-ahead traps and how the code handles them

| # | Trap | Handling in code |
|---|---|---|
| 1 | A 10-K for fiscal year *t* is often not public until months later | Text is indexed by EDGAR **filing date** and becomes usable at the end of the month *after* filing. A filing stays live until the next 10-K is available, and never for more than 15 months |
| 2 | Start-up databases are back-filled | This cannot be fully fixed with a snapshot database. Funding is used by calendar year with a one-year lag (see #7), and the limitation is disclosed |
| 3 | Pooled standardisation leaks the full-sample distribution | Scores are rank-transformed **within each month** only |
| 4 | Full-sample portfolio breakpoints | Quintiles are formed on the **current month's** cross-section |
| 5 | Vocabulary from the future | IDF and the neighbour pool for year *Y* use only start-ups funded by *Y*−1, and 10-Ks are projected into that fixed space |
| 6 | Survivorship on the public side | Point-in-time universe and stock–filing links, with delisting returns compounded in |
| 7 | Annual funding totals are not known until year-end | Every month in year *Y* uses funding from *Y*−1 |
| 8 | Start-up descriptions are a recent snapshot (pivots rewrite history) | There is no fix with any snapshot source. This is disclosed as a limitation |

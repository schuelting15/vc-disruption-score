#!/usr/bin/env python3
"""
THE FULL RUN.  One command, start to finish.

    python scripts/make_synthetic_data.py            # or point DS_DATA at the licensed data
    python src/full_run.py

Rebuilds the 10-K text table from the cached Item 1 extracts, computes the
Disruption Score, joins CRSP, writes the estimation panel, and runs the whole
modelling ladder with placebos and a block bootstrap.

Inputs are read from $DS_DATA (default: ./data), with the layout described in
docs/DATA.md. The errno-35 guard below dates from when the cache lived on OneDrive.

Specification is fixed (pre-registered 17 September 2026, see docs/METHOD.md). Do not tune it here.
"""
import os, re, sys, json, gzip, time, glob
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.environ.get("DS_DATA", os.path.join(ROOT, "data"))
INT   = os.path.join(DATA, "interim")
CACHE = os.path.join(DATA, "raw/edgar")
CRSP  = os.path.join(DATA, "raw/crsp")
PROC  = os.path.join(DATA, "processed")
OUT   = os.environ.get("DS_OUT", os.path.join(ROOT, "results"))
TAB   = os.path.join(OUT, "tables")
for d in (PROC, TAB): os.makedirs(d, exist_ok=True)

K_MAX, SIM_MIN, BLOCK = 100, 0.10, 200      # frozen -- DECISIONS.md 17 Sep
PANEL_YEARS = range(2014, 2025)
PRIMARY = "ds_level_raw"                    # frozen -- DECISIONS.md 17 Sep

L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))
def save_log():
    open(os.path.join(OUT, "full_run_report.txt"), "w").write("\n".join(L))

STOP = set("""the a an and or of to in for is are was were be been being with that this these those it its
on by as at from we our their his her which when who whom whose you your they them he she i me my
has have had do does did will would can could should may might must shall not no nor but if then than
also such other others more most some any all each every both few many much very into over under
between through during before after above below up down out off again further once here there where how
company companies product products service services platform solution solutions offering offerings
provides provide providing offers offer main focuses focus specializes specializing primary
designed design develops develop developing includes include including based use uses using utilizes
allows enables enabling aims aim business customers clients users market markets industry industries
inc corp corporation llc ltd plc group holdings subsidiaries subsidiary common stock shares annual
report fiscal year years ended december january february march april june july august september
october november million billion approximately per cent percent under pursuant thereof herein thereto
respectively item part page table contents form securities exchange commission act registrant""".split())
WORD = re.compile(r"[a-z][a-z-]{2,}")
def tok(s): return [w for w in WORD.findall(str(s).lower()) if w not in STOP]

def nw_se(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 3: return np.nan
    lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = x - x.mean(); v = (e @ e) / T
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / T
    return np.sqrt(max(v, 0) / T)
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / nw_se(x) if len(x) > 2 else np.nan

t0 = time.time()
log("=" * 74); log("DISRUPTION SCORE -- FULL RUN"); log("=" * 74)

# ══════════════════ 1. text, straight from the cache ══════════════════
man = {}
with open(os.path.join(CACHE, "_manifest.jsonl")) as fh:
    for line in fh:
        try: r = json.loads(line)
        except Exception: continue
        man[f'{r["cik"]}_{r["accession"].replace("-","")}'] = r["filing_date"]

names = sorted(f for f in os.listdir(CACHE) if f.endswith(".item1.txt.gz"))
log(f"\ncached Item 1 extracts : {len(names):,}")
rows, n_lock, n_bad = [], 0, []
for i, f in enumerate(names):
    stem = f[:-len(".item1.txt.gz")]
    if stem not in man: continue
    try:
        with gzip.open(os.path.join(CACHE, f), "rt", errors="ignore") as fh:
            method = fh.readline().strip(); text = fh.read()
    except OSError as e:
        n_bad.append(f)
        n_lock += 1
        if n_lock == 40:
            log("\n" + "!" * 74)
            log("STOPPING: OneDrive has dehydrated the cache -- 40 files in a row")
            log("failed with errno 35. Right-click the 'Big Data' folder in Finder,")
            log("choose 'Always Keep on This Device', wait for the green ticks, and")
            log("run this again. Nothing is broken; the files are just not on disk.")
            log("!" * 74); save_log(); sys.exit(1)
        continue
    n_lock = 0
    fd = man[stem]
    am = pd.Timestamp(fd) + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)  # TRAP #1: end of the month after filing
    # (MonthEnd(0) first: MonthEnd(1)+MonthEnd(1) made a filing ON a month-end wait two months; fixed 28 Sep 2026)
    rows.append(dict(cik=f"{int(stem.split('_')[0]):010d}", accession=stem.split("_")[1],
                     filing_date=fd, available_month=am, path=os.path.join(CACHE, f),
                     n_chars=len(text), extract_method=method))
    del text   # memory: texts are streamed into TF-IDF later, not held (18 Sep)
    if (i + 1) % 4000 == 0:
        log(f"   read {i+1:,}/{len(names):,}  [{time.time()-t0:.0f}s]")

# Guard added 18 Sep: zero-filled OneDrive placeholders raise BadGzipFile (an
# OSError) and used to be skipped silently -- a run on 70% of the corpus that
# looks normal. Refuse to continue unless every cached file was readable.
if n_bad and not os.environ.get("ALLOW_UNREADABLE"):
    log(f"\nSTOPPING: {len(n_bad):,} cached files are unreadable, e.g. {n_bad[:3]}")
    log("Re-hydrate (00_scripts/hydrate_edgar.sh) and re-bundle. "
        "Set ALLOW_UNREADABLE=1 only for a deliberate partial run.")
    save_log(); sys.exit(2)
log(f"unreadable files       : {len(n_bad):,}")
pub = pd.DataFrame(rows)
pub.drop(columns=["path"]).to_parquet(INT + "/filing_index.parquet", index=False)
n_all = len(pub)
pub = pub.sort_values(["cik", "available_month"]).reset_index(drop=True)
# TRAP #1: a 10-K is live until the firm's NEXT 10-K is available -- the next one it FILED,
# whether or not its Item 1 could be extracted. So next_av is set before failed extractions
# are dropped; otherwise an old description stayed live for years behind a failed one
# (1,006 stock-months; code review, 27 Sep 2026).
pub["next_av"] = pub.groupby("cik").available_month.shift(-1)
# ...and never more than 15 months after it became available: a firm that stops filing is not
# scored on an old description (28 Sep 2026, leakage audit T5).
pub["expires"] = pub.available_month + pd.offsets.MonthEnd(15)
pub = pub[pub.n_chars >= 1000].reset_index(drop=True)
def is_live(m):
    return (pub.available_month <= m) & (pub.next_av.isna() | (pub.next_av > m)) & (pub.expires >= m)
log(f"\nfilings read           : {n_all:,}")
log(f"usable (>=1,000 chars) : {len(pub):,}   firms: {pub.cik.nunique():,}")
log(f"median Item 1 length   : {pub.n_chars.median():,.0f} chars")
log("\nextraction methods:")
log(pub.extract_method.value_counts().to_string())

# ══════════════════ 2-3. word space and neighbours, POINT IN TIME (TRAP #5) ══════════════════
# For months in year Y the word space (IDF, fitted on startups only) and the k nearest startups
# come only from startups first funded by Y-1. A pool of every
# startup known in 2026 matched firms in 2014 to startups founded later (62% of the gated slots in
# 2014), so early scores used how crowded a space would become. Fixed 28 Sep 2026 (leakage audit
# T5); k, the similarity gate and the four variants are unchanged.
prv = pd.read_parquet(INT + "/startups.parquet")
prv = prv[~prv.flag_junk_description].reset_index(drop=True)
fy  = pd.read_parquet(INT + "/funding_by_year.parquet")
# A startup joins the pool the year after its FIRST FUNDED year. Not its founding year: Harmonic
# holds only startups with a Seed-D round in 2012-25, so "founded by Y-1" admitted startups because
# they raised money later (audit T5 round 2, 28 Sep 2026).
born = prv.startup_id.map(fy.groupby("startup_id").year.min()).fillna(9999).to_numpy(float)
log(f"\nprivate universe       : {len(prv):,} startups; each joins the pool the year after its first round")
def _texts(paths):
    for p in paths:
        with gzip.open(p, "rt", errors="ignore") as fh:
            fh.readline(); yield fh.read()
NB = {}                      # year -> (pub rows live that year, their neighbours in prv, gated weights)
for Y in PANEL_YEARS:
    pool = np.flatnonzero(born <= Y - 1)
    vec = TfidfVectorizer(tokenizer=tok, min_df=5, sublinear_tf=True, token_pattern=None)
    Xp = normalize(vec.fit_transform(prv.description.fillna("").iloc[pool]))
    rows_y = np.flatnonzero(np.logical_or.reduce(
        [is_live(m).to_numpy() for m in pd.date_range(f"{Y}-01-31", f"{Y}-12-31", freq="ME")]))
    Xk = normalize(vec.transform(_texts(pub.path.to_numpy()[rows_y])))
    K = min(K_MAX, Xp.shape[0] - 1)
    nbr_i = np.zeros((len(rows_y), K), np.int32); nbr_s = np.zeros((len(rows_y), K), np.float32)
    for b in range(0, len(rows_y), BLOCK):
        sl = slice(b, min(b + BLOCK, len(rows_y)))
        S = (Xk[sl] @ Xp.T).toarray()
        pa = np.argpartition(-S, K, axis=1)[:, :K]
        r = np.arange(S.shape[0])[:, None]
        v = S[r, pa]; o = np.argsort(-v, axis=1)
        nbr_i[sl] = pool[pa[r, o]]; nbr_s[sl] = v[r, o]
    W = np.where(nbr_s >= SIM_MIN, nbr_s, 0.0).astype(np.float64)
    NB[Y] = (rows_y, nbr_i, W)
    log(f"   {Y}: pool {len(pool):>6,} startups, vocabulary {Xp.shape[1]:>6,}, {len(rows_y):,} 10-Ks live, "
        f"above the {SIM_MIN} gate: median {np.median((W > 0).sum(1)):.0f} of {K}  [{time.time()-t0:.0f}s]")
pub = pub.drop(columns=["path"])

# ══════════════════ 4. funding matrix ══════════════════
YEARS = list(range(2012, 2026)); YC = {y: i for i, y in enumerate(YEARS)}
s2r = {s: i for i, s in enumerate(prv.startup_id)}
F_usd = np.zeros((len(prv), len(YEARS))); F_cnt = np.zeros((len(prv), len(YEARS)))
for sid, yr, n, amt in zip(*[fy[fy.startup_id.isin(s2r)][c] for c in
                             ("startup_id", "year", "n_rounds", "amount_usd")]):
    r_, c_ = s2r[sid], YC[yr]
    F_cnt[r_, c_] += n
    if not pd.isna(amt): F_usd[r_, c_] += amt

# ══════════════════ 5. DS by firm-month ══════════════════
months = pd.date_range(f"{min(PANEL_YEARS)}-01-31", f"{max(PANEL_YEARS)}-12-31", freq="ME")
chunks, n_capped = [], 0
for m in months:
    n_capped += int(((pub.available_month <= m) & (pub.next_av.isna() | (pub.next_av > m))
                     & (pub.expires < m)).sum())
    live = pub[is_live(m)]
    if not len(live): continue
    rows_y, nbr_y, W_y = NB[m.year]                          # that year's pool (TRAP #5)
    pos = np.searchsorted(rows_y, live.index.to_numpy())
    assert (rows_y[pos] == live.index.to_numpy()).all()
    nbr, w = nbr_y[pos], W_y[pos]
    c = YC[m.year - 1]                                       # TRAP #7
    f_usd, f_cnt = F_usd[nbr, c], F_cnt[nbr, c]
    prev = F_usd[nbr, YC[m.year - 2]] if m.year - 2 >= YEARS[0] else None
    lvl = (w * f_usd).sum(1)
    chunks.append(pd.DataFrame(dict(
        cik=live.cik.values, month=m, live_from=live.available_month.values, ds_level_raw=lvl,
        ds_change_raw=(lvl - (w * prev).sum(1)) if prev is not None else np.nan,
        ds_count=(w * f_cnt).sum(1), n_neighbours=(w > 0).sum(1),
        mean_sim=w.sum(1) / np.maximum((w > 0).sum(1), 1))))
DS = pd.concat(chunks, ignore_index=True)
DS.to_parquet(PROC + "/disruption_score.parquet", index=False)
log(f"\nfirm-months scored     : {len(DS):,}  ({DS.cik.nunique():,} firms); "
    f"{n_capped:,} filer-months past the 15-month cap on 10-K age (all filers, before the stock "
    f"link; mostly firms that stopped filing and trading)")

# ══════════════════ 6. CRSP ══════════════════
use = ["PERMNO", "date", "SHRCD", "EXCHCD", "PRC", "SHROUT", "RET", "DLRET", "SICCD"]
cr = pd.read_csv(sorted(glob.glob(CRSP + "/crsp_msf_*.csv.gz"))[-1], usecols=use,
                 dtype={"PERMNO": "int32"}, parse_dates=["date"], low_memory=False)
cr = cr[cr.SHRCD.isin([10, 11]) & cr.EXCHCD.isin([1, 2, 3])].copy()
for c in ["RET", "DLRET"]: cr[c] = pd.to_numeric(cr[c], errors="coerce")
cr["ret_adj"] = np.where(cr.RET.notna() & cr.DLRET.notna(),      # TRAP #6
                         (1 + cr.RET) * (1 + cr.DLRET) - 1,
                         np.where(cr.RET.notna(), cr.RET, cr.DLRET))
cr["mktcap"] = cr.PRC.abs() * cr.SHROUT
# Industry: Fama-French 12 from CRSP SICCD as of that month (point-in-time).
# Pre-registered 18 Sep 2026, before the full corpus was scored. ROBUSTNESS ONLY.
_FF12 = [(1,[(100,999),(2000,2399),(2700,2749),(2770,2799),(3100,3199),(3940,3989)]),
 (2,[(2500,2519),(2590,2599),(3630,3659),(3710,3711),(3714,3714),(3716,3716),(3750,3751),
     (3792,3792),(3900,3939),(3990,3999)]),
 (3,[(2520,2589),(2600,2699),(2750,2769),(3000,3099),(3200,3569),(3580,3629),(3700,3709),
     (3712,3713),(3715,3715),(3717,3749),(3752,3791),(3793,3799),(3830,3839),(3860,3899)]),
 (4,[(1200,1399),(2900,2999)]), (5,[(2800,2829),(2840,2899)]),
 (6,[(3570,3579),(3660,3692),(3694,3699),(3810,3829),(7370,7379)]),
 (7,[(4800,4899)]), (8,[(4900,4949)]), (9,[(5000,5999),(7200,7299),(7600,7699)]),
 (10,[(2830,2839),(3693,3693),(3840,3859),(8000,8099)]), (11,[(6000,6999)])]
_sic = pd.to_numeric(cr.SICCD, errors="coerce")
cr["ff12"] = np.where(_sic.notna(), 12, np.nan)
for _k, _rs in _FF12:
    for _a, _b in _rs:
        cr.loc[_sic.between(_a, _b), "ff12"] = _k
cr["month"] = cr.date + pd.offsets.MonthEnd(0)
# The WRDS extract repeats 1,822 stock-months as identical rows. Shifting before dropping them
# made "next month" the SAME month for those rows, and a gap in a stock's history made it a
# later month. Both fixed 23 Sep 2026 (01_docs/DECISIONS.md).
cr = cr.drop_duplicates(["PERMNO", "month"]).sort_values(["PERMNO", "month"])
nxt = cr.groupby("PERMNO").month.shift(-1)
cr["ret_fwd_1m"] = cr.groupby("PERMNO").ret_adj.shift(-1).where(
    nxt == cr.month + pd.offsets.MonthEnd(1))
log(f"CRSP rows kept         : {len(cr):,}  (delisting returns {cr.DLRET.notna().sum():,})")

# Links are POINT IN TIME: one row per (PERMNO, universe year, cik), written by 21_fill_links.py
# from 20_build_universe.py's CUSIP links, the reviewed exclusions (a hand-reviewed exclusion list,
# e.g. old TCF's CIK on Chemical Financial's stock) and the verified fills. A 10-K scores a stock
# only in the universe years its link covers, so no CIK carries a stale or another company's
# 10-K into a year it was not verified for (27 Sep 2026). Universe year Y = July Y to June Y+1.
if not os.path.exists(INT + "/universe_links.csv"):
    sys.exit("universe_links.csv missing: build the point-in-time links first (docs/DATA.md)")
links = pd.read_csv(INT + "/universe_links.csv", dtype={"cik": str})
links = links.assign(cik=links.cik.str.zfill(10)).rename(columns={"PERMNO": "permno", "year": "rank_year"})
DS["rank_year"] = np.where(DS.month.dt.month >= 7, DS.month.dt.year, DS.month.dt.year - 1)
DS = DS.merge(links[["cik", "rank_year", "permno", "firm", "ticker"]], on=["cik", "rank_year"], how="inner")
P = DS.merge(cr[["PERMNO", "month", "ret_adj", "ret_fwd_1m", "mktcap", "ff12"]],
             left_on=["permno", "month"], right_on=["PERMNO", "month"], how="inner")
# One CIK per stock-year by construction; kept as a guard: if two CIKs ever score one
# stock-month, keep the newest live 10-K (on a tie, the higher CIK, i.e. the successor).
P = P.sort_values(["permno", "month", "live_from", "cik"]).drop_duplicates(["permno", "month"], keep="last")
uni = pd.read_csv(INT + "/universe_permno_year.csv")[["year", "PERMNO"]].drop_duplicates()
uni["in_universe"] = True
P["rank_year"] = np.where(P.month.dt.month >= 7, P.month.dt.year, P.month.dt.year - 1)
P = P.merge(uni, left_on=["rank_year", "permno"], right_on=["year", "PERMNO"],
            how="left", suffixes=("", "_u"))
P = P[P.in_universe.fillna(False)].copy()
for c in ["ds_level_raw", "ds_change_raw", "ds_count"]:
    P[c + "_rank"] = P.groupby("month")[c].rank(pct=True) - 0.5      # TRAP #3
P["ds_level_scaled"] = P.ds_level_raw / P.mktcap.replace(0, np.nan)
P["lnME"] = np.log(P.mktcap.replace(0, np.nan))
P = P.sort_values(["month", "permno"]).reset_index(drop=True)
P.to_parquet(PROC + "/panel.parquet", index=False)
P.rename(columns={"permno": "id", PRIMARY: "score", "ff12": "industry"})[
    ["id", "month", "score", "mktcap", "ret_fwd_1m", "industry", "lnME"]].dropna(
    subset=["score", "ret_fwd_1m", "mktcap"]).to_csv(TAB + "/panel_for_R.csv", index=False)

g = P.groupby("month").size()
log("\n" + "=" * 74); log("PANEL"); log("=" * 74)
log(f"rows {len(P):,} | firms {P.permno.nunique():,} | months {P.month.nunique()} "
    f"({P.month.min():%Y-%m} to {P.month.max():%Y-%m}) | median {g.median():.0f} firms/month")

# ══════════════════ 7. the tests ══════════════════
def sort_ls(d, col, n_bins=5, cost_bps=20):
    d = d[np.isfinite(d[col]) & d.ret_fwd_1m.notna() & (d.mktcap > 0)]
    prev, res = {}, []
    for m, gg in d.groupby("month", sort=True):
        if len(gg) < 50: continue
        q = pd.qcut(gg[col].rank(method="first"), n_bins, labels=False, duplicates="drop")
        if pd.Series(q).nunique() < n_bins: continue
        lo, hi = gg[q == 0], gg[q == n_bins - 1]
        if len(lo) < 5 or len(hi) < 5: continue
        wl, wh = lo.mktcap / lo.mktcap.sum(), hi.mktcap / hi.mktcap.sum()
        w = {**{i: -v for i, v in zip(lo.permno, wl)}, **{i: v for i, v in zip(hi.permno, wh)}}
        turn = sum(abs(w.get(i, 0) - prev.get(i, 0)) for i in set(w) | set(prev)) / 2
        prev = w
        res.append(dict(month=m, lo=(wl*lo.ret_fwd_1m).sum(), hi=(wh*hi.ret_fwd_1m).sum(),
                        turnover=turn))
    r = pd.DataFrame(res)
    if len(r):
        r["ls"] = r.hi - r.lo; r["ls_net"] = r.ls - r.turnover * cost_bps / 1e4
    return r

def fm_series(d, col, extra=(), shuffle=False, rng=None):
    out = []
    for _, gg in d.groupby("month", sort=True):
        gg = gg[gg.ret_fwd_1m.notna()]
        if len(gg) < 50: continue
        v = gg[col].values.copy()
        if shuffle: v = rng.permutation(v)
        X = [pd.Series(v).rank(pct=True).values - 0.5] + \
            [gg[e].rank(pct=True).values - 0.5 for e in extra]
        X = np.column_stack(X); ok = np.isfinite(X).all(1)
        if ok.sum() < 50: continue
        y = gg.ret_fwd_1m.values[ok]
        a, b = np.quantile(y, [.01, .99]); y = np.clip(y, a, b)
        out.append(np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), X[ok]]), y,
                                   rcond=None)[0][1])
    return np.array(out)

log("\n" + "=" * 74); log("RUNG 1 -- value-weighted quintile sorts"); log("=" * 74)
for col, nm in [("ds_count", "count of entrants"), ("ds_level_raw", "dollars raised"),
                ("ds_change_raw", "change in dollars"), ("ds_level_scaled", "dollars / mktcap")]:
    r = sort_ls(P, col)
    if not len(r): log(f"\n{nm}: too few firms"); continue
    log(f"\n{nm}   ({len(r)} months)")
    log(f"   gross {r.ls.mean()*100:+.3f} %/mo  t(NW) {tstat(r.ls):+.2f}   "
        f"net {r.ls_net.mean()*100:+.3f} %/mo  t(NW) {tstat(r.ls_net):+.2f}   "
        f"turnover {r.turnover.mean():.2f}")
    if col == PRIMARY: r.to_csv(TAB + "/ls_series.csv", index=False)

log("\n" + "=" * 74); log("RUNG 2 -- Fama-MacBeth"); log("=" * 74)
for col in ["ds_count", "ds_level_raw"]:
    for extra, lab in [((), "univariate"), (("lnME",), "with size control")]:
        s = fm_series(P, col, extra)
        log(f"   {col:<15} {lab:<18} {s.mean()*100:+.4f} %/mo   t(NW) {tstat(s):+.2f}"
            f"   ({len(s)} months)")

log("\n" + "=" * 74); log("PLACEBO + BOOTSTRAP + SUBPERIODS"); log("=" * 74)
rng = np.random.default_rng(11)
res = {}
for col in ["ds_count", "ds_level_raw"]:
    real = fm_series(P, col); t_real = tstat(real)
    ts = np.array([tstat(fm_series(P, col, shuffle=True, rng=rng)) for _ in range(200)])
    pval = float((np.abs(ts) >= abs(t_real)).mean())
    # Circular 12-month blocks, exactly T months per draw. Fixed 23 Sep 2026: blocks used to
    # stop at the sample end, so most draws were shorter than T and every interval too wide.
    T, B = len(real), 12
    boot = lambda x, n: np.array([x[(rng.integers(0, n, -(-n // B))[:, None] + np.arange(B)).ravel()
                                    % n][:n].mean() for _ in range(2000)]) * 100
    draws = boot(real, T)
    lo, hi = np.quantile(draws, [.025, .975])
    res[col] = dict(mean=float(real.mean()*100), t=float(t_real), placebo_p=pval,
                    boot_lo=float(lo), boot_hi=float(hi))
    log(f"\n{col}")
    log(f"   estimate        {real.mean()*100:+.3f} %/mo   t(NW) {t_real:+.2f}")
    log(f"   placebo p       {pval:.3f}   {'PASS' if pval < .05 else 'FAIL'}")
    log(f"   bootstrap 95%   [{lo:+.3f}, {hi:+.3f}]   "
        f"{'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES ZERO'}")
    # split at the median firm-month, as pre-registered; it is NOT mid-2019 (the panel grows),
    # so the halves are labelled by their actual months (labels fixed 23 Sep 2026)
    mid = P.month.quantile(.5)
    for lab, d in [(f"{P.month.min():%Y-%m} to {mid:%Y-%m}", P[P.month <= mid]),
                   (f"{mid + pd.offsets.MonthEnd(1):%Y-%m} on   ", P[P.month > mid])]:
        s = fm_series(d, col)
        log(f"   {lab}  {s.mean()*100:+.3f} %/mo   t(NW) {tstat(s):+.2f}")
    if col == PRIMARY:      # the pre-registered primary carries a size control (17 Sep, item 1)
        real = fm_series(P, col, ("lnME",)); t_real = tstat(real)
        ts = np.array([tstat(fm_series(P, col, ("lnME",), shuffle=True, rng=rng)) for _ in range(200)])
        lo, hi = np.quantile(boot(real, len(real)), [.025, .975])
        res[col].update(size_mean=float(real.mean()*100), size_t=float(t_real),
                        size_placebo_p=float((np.abs(ts) >= abs(t_real)).mean()),
                        size_boot_lo=float(lo), size_boot_hi=float(hi))
        log(f"   + size control  {real.mean()*100:+.3f} %/mo   t(NW) {t_real:+.2f}   placebo p "
            f"{res[col]['size_placebo_p']:.3f}   bootstrap 95% [{lo:+.3f}, {hi:+.3f}]")

log("\n" + "=" * 74); log("RUNG 3 -- industry-neutral (FF12, robustness; pre-registered 18 Sep)")
log("=" * 74)
log(f"   FF12 coverage {P.ff12.notna().mean():.1%}   firms per industry (median month):")
log("   " + "  ".join(f"{int(k)}:{v:.0f}" for k, v in
    P.groupby(["month", "ff12"]).size().groupby("ff12").median().items()))
def fm_ind(d, col):
    """FM with FF12 dummies: DS rank within month, industry fixed effects."""
    out = []
    for _, gg in d.groupby("month", sort=True):
        gg = gg[gg.ret_fwd_1m.notna() & gg.ff12.notna() & np.isfinite(gg[col])]
        if len(gg) < 50: continue
        x = gg[col].rank(pct=True).values - 0.5
        D = pd.get_dummies(gg.ff12.astype(int)).values.astype(float)
        y = gg.ret_fwd_1m.values; a, b = np.quantile(y, [.01, .99]); y = np.clip(y, a, b)
        out.append(np.linalg.lstsq(np.column_stack([x, D]), y, rcond=None)[0][0])
    return np.array(out)
Pi = P[P.ff12.notna()].copy()
for col in ["ds_count", "ds_level_raw"]:
    s = fm_ind(Pi, col)
    log(f"   {col:<15} FM + FF12 dummies  {s.mean()*100:+.4f} %/mo   t(NW) {tstat(s):+.2f}"
        f"   ({len(s)} months)")
    res.setdefault(col, {})["fm_ff12"] = float(s.mean()*100); res[col]["t_fm_ff12"] = float(tstat(s))
    # sort on the within-industry rank: quintiles are industry-balanced by construction
    Pi[col + "_indrank"] = Pi.groupby(["month", "ff12"])[col].rank(pct=True)
    r = sort_ls(Pi, col + "_indrank")
    if len(r):
        log(f"   {col:<15} within-FF12 sort   gross {r.ls.mean()*100:+.3f} %/mo  "
            f"t(NW) {tstat(r.ls):+.2f}   net {r.ls_net.mean()*100:+.3f}  t {tstat(r.ls_net):+.2f}")
log("   top-quintile industry mix, primary score (share of firm-months):")
Pq = P[P.ds_level_raw.notna()].copy()
Pq["q"] = Pq.groupby("month").ds_level_raw.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
mix = Pq[Pq.q == 4].ff12.value_counts(normalize=True).sort_index()
log("   " + "  ".join(f"{int(k)}:{v:.0%}" for k, v in mix.items()))

json.dump(res, open(TAB + "/full_run_summary.json", "w"), indent=1)
r = res[PRIMARY]
log(f"\nPRIMARY RESULT ({PRIMARY} with size control, per DECISIONS_2026-09-17 item 1): "
    f"{r['size_mean']:+.3f} %/mo, t = {r['size_t']:+.2f}, bootstrap [{r['size_boot_lo']:+.3f}, "
    f"{r['size_boot_hi']:+.3f}], placebo p = {r['size_placebo_p']:.3f}   "
    f"(univariate {r['mean']:+.3f}, t = {r['t']:+.2f})")
log(f"\ntotal runtime {time.time()-t0:.0f}s")
save_log()
print(f"\nwrote {OUT}/full_run_report.txt and {TAB}/")

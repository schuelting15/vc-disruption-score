#!/usr/bin/env python3
"""
Synthetic stand-in for the licensed inputs of the Disruption Score pipeline.

The real study uses CRSP/Compustat (WRDS licence), a Harmonic startup export (commercial
licence) and SEC 10-K text. None of that may be redistributed, so this script writes
fake data with exactly the same files, columns and formats, so that `src/full_run.py`
runs end to end on a laptop in a few minutes.

Nothing here is real: companies, texts, funding rounds and returns are all drawn from
the random process below. The returns contain a *planted* effect of known size, so the
run doubles as a check that the pipeline recovers a signal when one exists (and reports
a null when `--effect 0`).

    python scripts/make_synthetic_data.py                 # default: planted effect
    python scripts/make_synthetic_data.py --effect 0      # pure-noise control run

Files written under data/ (git-ignored):
    raw/edgar/_manifest.jsonl, raw/edgar/<cik>_<acc>.item1.txt.gz   (10-K Item 1 cache)
    interim/startups.parquet, interim/funding_by_year.parquet      (startup corpus)
    raw/crsp/crsp_msf_synthetic.csv.gz                              (monthly stock file)
    interim/universe_links.csv, interim/universe_permno_year.csv   (point-in-time links)
"""
import argparse, gzip, json, os, shutil
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Twelve product spaces. Each has its own vocabulary and an SIC code that maps into a
# different Fama-French-12 industry, so the industry-neutral tests have something to do.
SECTORS = {
    "fintech":    (6199, "payments lending banking credit wallet ledger invoicing remittance brokerage underwriting fraud compliance deposits loans merchant checkout treasury crypto stablecoin mortgage insurance claims card interchange settlement"),
    "biotech":    (2836, "antibody oncology therapeutics clinical trial molecule genomics gene cell biologics drug receptor inhibitor vaccine rna enzyme assay biomarker tumor dosing preclinical indication fda antigen peptide"),
    "medtech":    (3841, "device implant surgical catheter imaging diagnostic monitoring wearable patient hospital cardiac orthopedic stent ultrasound endoscopy glucose infusion telehealth physician clinic reimbursement sterile robotic-surgery cardiology"),
    "software":   (7372, "saas cloud enterprise workflow analytics database api developer security identity observability devops crm erp subscription collaboration integration deployment kubernetes encryption endpoint dashboard low-code machine-learning"),
    "semis":      (3674, "semiconductor chip wafer foundry silicon lithography transistor processor gpu memory dram fabrication analog photonics asic fpga substrate yield accelerator interconnect power-management die epitaxy etching"),
    "cleantech":  (4911, "solar wind battery storage grid hydrogen electrolyzer renewable carbon emissions photovoltaic inverter turbine utility charging decarbonization geothermal heat-pump capture biofuel nuclear transmission microgrid offset"),
    "mobility":   (3711, "vehicle electric autonomous fleet ride-hailing scooter lidar drivetrain automotive dealer trucking routing parking telematics chassis motor aftermarket navigation freight last-mile transit driver cargo"),
    "ecommerce":  (5961, "retail shopping marketplace apparel fashion store brand ecommerce fulfillment warehouse returns loyalty grocery catalog merchandise direct-to-consumer cart inventory omnichannel coupon beauty footwear furniture resale"),
    "telecom":    (4813, "wireless spectrum broadband fiber carrier tower satellite bandwidth voip roaming subscriber backhaul router antenna cable latency connectivity isp messaging telecom operator handset prepaid postpaid"),
    "industrial": (3560, "manufacturing robotics machinery factory sensors maintenance predictive cnc additive printing welding assembly conveyor actuator pumps valves aerospace defense drone inspection construction tooling hydraulics compressor"),
    "foodtech":   (2000, "food beverage restaurant meal plant-based dairy snack nutrition agriculture farm crop seed fertilizer vertical-farming kitchen recipe organic coffee brewery alcohol ingredients flavor bakery meat poultry"),
    "materials":  (2821, "chemical polymer resin coating specialty catalyst plastic recycling composite adhesive steel aluminum mining lithium cement glass textile paint solvent pigment metallurgy alloy ceramics membrane"),
}
GENERIC = ("software data technology team global digital network scale growth partners revenue innovative "
           "leading efficient quality cost operations management process systems information research "
           "development value sales intelligence mobile online real-time secure flexible consumers people "
           "worldwide experience").split()
BOILER = ("competition employees regulation intellectual property patents seasonality suppliers risk segments "
          "revenue backlog headquarters trademarks distribution manufacturing pricing contracts governmental "
          "environmental licensing research").split()
YEARS = list(range(2012, 2026))


def text(rng, primary, secondary, n, w_pri, w_sec, extra=()):
    vp, vs = SECTORS[primary][1].split(), SECTORS[secondary][1].split()
    pools = [vp, vs, GENERIC, list(extra) or GENERIC]
    w = np.array([w_pri, w_sec, 1 - w_pri - w_sec, 0.0 if not extra else 0.15])
    w = w / w.sum()
    pick = rng.choice(len(pools), size=n, p=w)
    return " ".join(pools[k][rng.integers(len(pools[k]))] for k in pick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--effect", type=float, default=-0.010,
                    help="planted monthly return per unit of exposure rank (default -0.010 = -1.0%%/mo "
                         "top-minus-bottom, the 'threat' sign); 0 = no effect")
    ap.add_argument("--startups", type=int, default=4000)
    ap.add_argument("--firms", type=int, default=700)
    ap.add_argument("--seed", type=int, default=453)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sec = list(SECTORS)
    S = len(sec)

    for sub in ("raw/edgar", "raw/crsp", "interim", "processed"):
        p = os.path.join(a.out, sub)
        if sub == "raw/edgar" and os.path.isdir(p): shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)

    # ── 1. sector "heat": log funding intensity per sector-year (random walk) ──
    heat = np.zeros((S, len(YEARS)))
    heat[:, 0] = rng.normal(0, 0.3, S)
    for t in range(1, len(YEARS)):
        heat[:, t] = heat[:, t - 1] + rng.normal(0.05, 0.35, S)

    # ── 2. startups and their funding rounds (Harmonic-like) ──
    yrs_first = np.array(YEARS[:-1])                       # first rounds 2012-2024
    per_year = np.linspace(0.6, 1.4, len(yrs_first)); per_year /= per_year.sum()
    rows, fund = [], []
    sid = 0
    for t, y in enumerate(yrs_first):
        n_y = int(round(a.startups * per_year[t]))
        p_s = np.exp(0.6 * heat[:, t]); p_s /= p_s.sum()  # hot sectors attract entrants
        for s in rng.choice(S, size=n_y, p=p_s):
            s2 = rng.integers(S) if rng.random() < 0.3 else s
            junk = rng.random() < 0.02
            desc = "stealth" if junk else text(rng, sec[s], sec[s2], int(rng.integers(25, 45)), 0.70, 0.10)
            rows.append(dict(startup_id=f"S{sid:06d}", name=f"Synthetic Startup {sid:05d}",
                             sector=sec[s], founded_year=int(y - rng.integers(0, 3)),
                             description=desc, flag_junk_description=junk))
            amt, yy = np.exp(rng.normal(np.log(3e6) + heat[s, t], 1.0)), y
            while yy <= 2025:
                fund.append(dict(startup_id=f"S{sid:06d}", year=int(yy), n_rounds=1,
                                 amount_usd=float(amt) if rng.random() > 0.08 else np.nan))
                gap = int(rng.integers(1, 4)); yy += gap
                if yy > 2025 or rng.random() > 0.55: break
                amt *= 2.5 * np.exp(heat[s, YEARS.index(min(yy, 2025))] - heat[s, t])
            sid += 1
    startups = pd.DataFrame(rows)
    fy = pd.DataFrame(fund).groupby(["startup_id", "year"], as_index=False).agg(
        n_rounds=("n_rounds", "sum"), amount_usd=("amount_usd", lambda v: v.sum(min_count=1)))
    startups.to_parquet(os.path.join(a.out, "interim/startups.parquet"), index=False)
    fy.to_parquet(os.path.join(a.out, "interim/funding_by_year.parquet"), index=False)

    # True exposure of a sector in year Y: funding raised in Y by startups of that sector, per
    # startup already in the pool. That is what a firm's 100 nearest startups average, so the
    # Disruption Score should track it, noisily.
    first = fy.groupby("startup_id").year.min()
    sec_of = startups.set_index("startup_id").sector
    E = np.zeros((S, len(YEARS)))
    for t, y in enumerate(YEARS):
        pool = first[first <= y].index
        f = fy[(fy.year == y) & fy.startup_id.isin(pool)]
        tot = f.assign(sector=f.startup_id.map(sec_of)).groupby("sector").amount_usd.sum()
        cnt = sec_of[pool].value_counts()
        for i, s in enumerate(sec):
            E[i, t] = np.log1p(tot.get(s, 0.0) / max(cnt.get(s, 1), 1))

    # ── 3. public firms: 10-K Item 1 texts, one per fiscal year ──
    F = a.firms
    f_sec = rng.integers(S, size=F)
    f_sec2 = np.where(rng.random(F) < 0.6, rng.integers(S, size=F), f_sec)
    w_pri = rng.uniform(0.35, 0.60, F)
    start = np.where(rng.random(F) < 0.25, rng.integers(2013, 2021, F), 2012)   # IPOs
    stop = np.where(rng.random(F) < 0.15, rng.integers(2016, 2025, F), 2026)    # delistings
    cik = rng.choice(np.arange(1_000_000, 1_999_999), size=F, replace=False)
    man = open(os.path.join(a.out, "raw/edgar/_manifest.jsonl"), "w")
    for i in range(F):
        for y in range(max(2013, start[i] + 1), min(stop[i], 2025) + 1):
            fd = pd.Timestamp(f"{y}-02-10") + pd.Timedelta(days=int(rng.integers(0, 50)))
            acc = f"0000{rng.integers(900000, 999999)}-{y % 100:02d}-{rng.integers(0, 999999):06d}"
            failed = rng.random() < 0.02
            body = "Item 1. Business (extraction failed)" if failed else \
                text(rng, sec[f_sec[i]], sec[f_sec2[i]], int(rng.integers(250, 420)), w_pri[i], 0.20, BOILER)
            stem = f"{cik[i]}_{acc.replace('-', '')}"
            with gzip.open(os.path.join(a.out, f"raw/edgar/{stem}.item1.txt.gz"), "wt") as fh:
                fh.write(("failed" if failed else "header_span") + "\n" + body)
            man.write(json.dumps(dict(cik=int(cik[i]), accession=acc, filing_date=f"{fd:%Y-%m-%d}",
                                      form="10-K")) + "\n")
    man.close()

    # ── 4. CRSP-style monthly stock file with a planted effect ──
    months = pd.date_range("2012-01-31", "2025-12-31", freq="ME")
    mkt = rng.normal(0.008, 0.045, len(months))
    sec_shock = rng.normal(0, 0.02, (len(months), S))
    # firm exposure in year Y (log scale), mixing its two product spaces like its text does
    expo = (w_pri[:, None] * E[f_sec] + 0.20 * E[f_sec2]) / (w_pri[:, None] + 0.20)
    prc = np.exp(rng.normal(3.2, 0.8, F)); shrout = np.exp(rng.normal(10.5, 1.2, F)).round()
    beta = rng.normal(1.0, 0.3, F)
    out = []
    for t, m in enumerate(months):
        live = np.flatnonzero((start <= m.year) & (m.year <= stop) & ~((m.year == stop) & (m.month > 6)))
        if not len(live): continue
        # exposure that the score built at the end of month t-1 would see: funding of year(t-1) - 1
        prev = months[t - 1] if t else m
        e = expo[live, YEARS.index(max(prev.year - 1, YEARS[0]))]
        r_e = (pd.Series(e).rank(pct=True).to_numpy() - 0.5) if len(live) > 1 else np.zeros(1)
        ret = beta[live] * mkt[t] + sec_shock[t, f_sec[live]] + rng.normal(0, 0.11, len(live)) + a.effect * r_e
        ret = np.maximum(ret, -0.95)
        prc[live] *= 1 + ret
        last = (m.year == stop[live]) & (m.month == 6)
        dl = np.where(last & (rng.random(len(live)) < 0.7), rng.normal(-0.25, 0.2, len(live)).clip(-1, 1), np.nan)
        for j, i in enumerate(live):
            out.append((10001 + i, f"{m:%Y-%m-%d}", 11, 1 + i % 3, round(prc[i], 4), shrout[i],
                        round(ret[j], 6), None if np.isnan(dl[j]) else round(dl[j], 6),
                        SECTORS[sec[f_sec[i]]][0]))
    cr = pd.DataFrame(out, columns=["PERMNO", "date", "SHRCD", "EXCHCD", "PRC", "SHROUT", "RET", "DLRET", "SICCD"])
    cr.to_csv(os.path.join(a.out, "raw/crsp/crsp_msf_synthetic.csv.gz"), index=False, compression="gzip")

    # ── 5. point-in-time links: universe year Y = July Y to June Y+1 ──
    lk = [(f"{cik[i]:010d}", 10001 + i, y, f"Synthetic Corp {i:04d}", f"SYN{i:04d}")
          for i in range(F) for y in range(2013, 2025) if start[i] <= y < stop[i]]
    lk = pd.DataFrame(lk, columns=["cik", "PERMNO", "year", "firm", "ticker"])
    lk.to_csv(os.path.join(a.out, "interim/universe_links.csv"), index=False)
    lk[["year", "PERMNO"]].to_csv(os.path.join(a.out, "interim/universe_permno_year.csv"), index=False)

    json.dump(dict(effect=a.effect, seed=a.seed, startups=len(startups), funded_rows=len(fy),
                   firms=F, filings=sum(1 for _ in open(os.path.join(a.out, "raw/edgar/_manifest.jsonl"))),
                   crsp_rows=len(cr)), open(os.path.join(a.out, "synthetic_meta.json"), "w"), indent=1)
    print(open(os.path.join(a.out, "synthetic_meta.json")).read())


if __name__ == "__main__":
    main()

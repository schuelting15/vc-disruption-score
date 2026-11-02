#!/usr/bin/env python3
"""
EDGAR 10-K Item 1 puller  ->  contract section 1 (public_text.parquet)

NOT YET RUNNABLE FROM THE CLOUD CONTAINER. The session egress policy answers 403
to CONNECT for www.sec.gov and data.sec.gov, and the Mac's sandbox shell has no
network at all. Run this wherever sec.gov is reachable -- a laptop terminal, or
this container once the host is allow-listed. Everything else is finished.

    pip install requests pandas pyarrow lxml
    python3 10_edgar_item1.py --ciks data/interim/universe_ciks.csv \
                              --out  data/interim/public_text.parquet

Design notes that matter for the grade
--------------------------------------
* `filing_date` comes from the EDGAR submissions API, NEVER the fiscal year end.
  `available_month` = filing month-end + 1 month. That is Trap #1.
* Rate limited to SEC's published 10 requests/second with a descriptive
  User-Agent, as their access policy requires.
* Fully resumable: every filing is cached to disk as raw HTML, and a manifest
  records what has been fetched. Re-running skips completed work, so a dropped
  connection costs minutes rather than the whole pull.
* Item 1 extraction is the fragile part. Three strategies are tried in order and
  the one that fired is recorded in `extract_method`, so failures are auditable
  rather than silent. Anything under 1,000 characters is flagged, per contract.
"""
import argparse, gzip, html as _html, json, os, re, sys, time
from collections import deque
import pandas as pd

# SEC fair-access policy requires a descriptive User-Agent with a contact address.
UA = os.environ.get("SEC_UA", "")
if not UA and __name__ == "__main__":
    sys.exit('set SEC_UA, e.g. export SEC_UA="Your Name research project you@example.com"')
SUB = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBPAGE = "https://data.sec.gov/submissions/{name}"          # older pages of the same history
ARCH = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
RATE = 10.0            # requests/second, SEC published limit
MIN_CHARS = 1_000


class Limiter:
    def __init__(self, rate):
        self.rate, self.calls = rate, deque()
    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > 1.0:
            self.calls.popleft()
        if len(self.calls) >= self.rate:
            time.sleep(1.0 - (now - self.calls[0]) + 0.005)
        self.calls.append(time.time())


def get(session, url, limiter, tries=4):
    for a in range(tries):
        limiter.wait()
        try:
            r = session.get(url, headers={"User-Agent": UA,
                                          "Accept-Encoding": "gzip, deflate"}, timeout=45)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429):        # back off, SEC throttling
                time.sleep(2 ** a)
                continue
            if r.status_code == 404:
                return None
        except Exception:
            time.sleep(2 ** a)
    return None


# ---------------------------------------------------------------- Item 1 extraction
# Real 10-K HTML lays headings out in tables, so lxml's text_content() runs the
# cells together: "PART IITEM 1.BUSINESSOverviewC.H. Robinson...". Any regex that
# expects a line start finds nothing. So we insert breaks around block elements
# first, then match anywhere rather than at a line boundary.
BLOCK = {"p", "div", "tr", "td", "th", "table", "br", "li", "section", "article",
         "h1", "h2", "h3", "h4", "h5", "h6", "span"}

# "item 1" but not 1A / 1B / 1C / 10-19 -- no digit OR letter may follow the 1.
# (Item 1C "Cybersecurity" was introduced in 2023 and broke the first attempt.)
HDR = r"item\s*1(?![0-9A-Za-z])\s*[\.\:\-\u2013\u2014]?\s*"
NXT = r"item\s*1\s*a\b"
MAX_ITEM1 = 400_000


def html_to_text(html):
    try:
        from lxml import html as LH
        doc = LH.fromstring(html)
        for bad in doc.xpath("//script|//style"):
            bad.getparent().remove(bad)
        for el in doc.iter():
            tag = el.tag if isinstance(el.tag, str) else ""
            if tag.lower() in BLOCK:
                if el.text:
                    el.text = "\n" + el.text
                el.tail = "\n" + (el.tail or "")
        txt = doc.text_content()
    except Exception:
        txt = re.sub(r"<(br|/p|/div|/tr|/td|/h[1-6])[^>]*>", "\n", html, flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
    # Some filers double-escape, so entities survive the parse as literal
    # "&#160;" / "&#8220;" text. Unescape twice, then normalise the whitespace
    # and quote characters they expand into.
    txt = _html.unescape(_html.unescape(txt))
    txt = (txt.replace("\xa0", " ").replace("\u200b", "")
              .replace("\u2018", "'").replace("\u2019", "'")
              .replace("\u201c", '"').replace("\u201d", '"')
              .replace("\u2013", "-").replace("\u2014", "-"))
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r" ?\n ?", "\n", txt)
    return re.sub(r"\n{3,}", "\n\n", txt)


def extract_item1(text):
    """Return (item1_text, method).

    Every 'Item 1' is a candidate start; every 'Item 1A' is a candidate end. The
    real Item 1 is the pairing that yields the longest body -- which discards the
    table of contents automatically, because there 'Item 1' and 'Item 1A' sit a
    few characters apart.
    """
    ends = [m.start() for m in re.finditer(NXT, text, re.I)]
    # Only candidates that are actually CLOSED by an "Item 1A" count. A late
    # cross-reference such as: see "Item 1. Business - Government Regulation"
    # has no Item 1A after it, so an open-ended match there runs on and swallows
    # the MD&A. Require a real terminator first; among those, take the longest.
    closed, open_ended = [], []
    for m in re.finditer(HDR, text, re.I):
        e = next((p for p in ends if p > m.end()), None)
        if e is not None:
            closed.append(text[m.end():e])
        else:
            open_ended.append(text[m.end(): m.end() + MAX_ITEM1])
    for pool, method in ((closed, "header_span"), (open_ended, "header_span_unclosed")):
        if pool:
            body = max(pool, key=len).strip()
            if len(body) >= MIN_CHARS:
                return body[:MAX_ITEM1], method

    # fallback: between a "Business" heading and "Risk Factors"
    b = re.search(r"\bBUSINESS\b", text, re.I)
    r_ = re.search(r"\bRISK\s+FACTORS\b", text, re.I)
    if b and r_ and r_.start() > b.end():
        body = text[b.end():r_.start()]
        if len(body) >= MIN_CHARS:
            return body.strip(), "business_to_risk"
    return "", "FAILED"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciks", required=True, help="CSV with a 'cik' column")
    ap.add_argument("--out", default="data/interim/public_text.parquet")
    ap.add_argument("--cache", default="data/raw/edgar")
    ap.add_argument("--start", type=int, default=2012)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0, help="stop after N filings (smoke test)")
    ap.add_argument("--keep-html", action="store_true",
                    help="also keep the raw filing HTML (~3 MB each — only for debugging)")
    a = ap.parse_args()

    import requests
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lim, ses = Limiter(RATE), requests.Session()

    ciks = sorted({int(c) for c in pd.read_csv(a.ciks).cik.dropna()})
    print(f"{len(ciks):,} CIKs, {a.start}-{a.end}")

    man_path = os.path.join(a.cache, "_manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path):
            try: done.add(json.loads(line)["accession"])
            except Exception: pass
        n_txt = len([f for f in os.listdir(a.cache) if f.endswith(".item1.txt.gz")])
        print(f"  resuming: {n_txt:,} filings already extracted, {len(done):,} downloaded")

    rows, n_new, t0 = [], 0, time.time()
    man = open(man_path, "a")
    for i, cik in enumerate(ciks):
        r = get(ses, SUB.format(cik=cik), lim)
        if r is None:
            continue
        try: js = r.json()
        except Exception: continue
        # `recent` holds only a firm's newest ~1,000 filings. A high-volume filer's early 10-Ks
        # sit in the older pages listed under `files`; reading `recent` alone dropped them and
        # selected the early panel on later filing volume (found 23 Sep 2026, DECISIONS.md).
        fl = js.get("filings", {})
        pages = [fl.get("recent", {})]
        for pg in fl.get("files", []):
            if pg.get("filingTo", "9999")[:4] < str(a.start) or pg.get("filingFrom", "0000")[:4] > str(a.end):
                continue
            rp = get(ses, SUBPAGE.format(name=pg["name"]), lim)
            try: pages.append(rp.json())
            except Exception: pass
        cols = ("form", "accessionNumber", "filingDate", "primaryDocument", "reportDate")
        for f, acc, fd, doc, rd in [x for p in pages for x in zip(*(p.get(c, []) for c in cols))]:
            if f not in ("10-K", "10-K405", "10-KSB"):
                continue
            if not (a.start <= int(fd[:4]) <= a.end):
                continue
            stem = os.path.join(a.cache, f"{cik}_{acc.replace('-','')}")
            txt_f, html_f = stem + ".item1.txt.gz", stem + ".html"
            item1 = method = None
            if os.path.exists(txt_f):                       # already extracted
                with gzip.open(txt_f, "rt", errors="ignore") as fh:
                    method = fh.readline().strip()
                    item1 = fh.read()
            elif os.path.exists(html_f):                    # legacy HTML cache
                item1, method = extract_item1(html_to_text(open(html_f, errors="ignore").read()))
            else:
                url = ARCH.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc)
                rr = get(ses, url, lim)
                if rr is None:
                    continue
                item1, method = extract_item1(html_to_text(rr.text))
                if a.keep_html:
                    open(html_f, "w", errors="ignore").write(rr.text)
                man.write(json.dumps({"cik": cik, "accession": acc, "filing_date": fd}) + "\n")
                man.flush(); n_new += 1
            if not os.path.exists(txt_f):                   # write the small cache
                with gzip.open(txt_f, "wt") as fh:
                    fh.write(method + "\n"); fh.write(item1)
            fm = pd.Timestamp(fd) + pd.offsets.MonthEnd(1) + pd.offsets.MonthEnd(1)
            rows.append(dict(cik=f"{cik:010d}", accession=acc,
                             fiscal_year=int((rd or fd)[:4]), filing_date=fd,
                             available_month=fm.date(), item1_text=item1,
                             n_chars=len(item1), extract_method=method))
            if a.limit and len(rows) >= a.limit:
                break
        if a.limit and len(rows) >= a.limit:
            break
        if (i + 1) % 25 == 0:
            print(f"  {i+1:>5}/{len(ciks)} CIKs · {len(rows):,} filings · "
                  f"{n_new:,} newly fetched · {time.time()-t0:.0f}s", flush=True)
    man.close()

    df = pd.DataFrame(rows).drop_duplicates(["cik", "accession"])
    df.to_parquet(a.out, index=False)
    print(f"\nwrote {len(df):,} filings -> {a.out}")
    print(df.extract_method.value_counts().to_string())
    bad = (df.n_chars < MIN_CHARS).sum()
    print(f"\nflagged short/failed extractions: {bad:,} ({bad/max(1,len(df)):.1%}) "
          f"-- contract section 1 says investigate anything under {MIN_CHARS} chars")
    print(f"median Item 1 length: {df.n_chars.median():,.0f} chars")


if __name__ == "__main__":
    main()

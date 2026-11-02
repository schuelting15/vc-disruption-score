# Data

No licensed data is in this repository:

- CRSP and Compustat (via WRDS) forbid redistribution.
- The Harmonic start-up export is commercially licensed.
- SEC filings are public domain, but the text cache is large and can be regenerated.

`scripts/make_synthetic_data.py` writes fake data in exactly the layout below. To run on real data, build the same files and set `DS_DATA` to their root folder.

## Layout expected by `src/full_run.py`

| File | Columns / format | Real source |
|---|---|---|
| `raw/edgar/_manifest.jsonl` | one JSON per filing: `cik` (int), `accession` (`0000000000-YY-NNNNNN`), `filing_date` (YYYY-MM-DD) | SEC EDGAR submissions API, via `src/edgar_item1.py` |
| `raw/edgar/<cik>_<accession without dashes>.item1.txt.gz` | first line: extraction method, then the Item 1 text | `src/edgar_item1.py` |
| `interim/startups.parquet` | `startup_id`, `description`, `flag_junk_description` (bool) | Harmonic company export |
| `interim/funding_by_year.parquet` | `startup_id`, `year`, `n_rounds`, `amount_usd` (NaN if undisclosed) | Harmonic funding rounds, aggregated to calendar years |
| `raw/crsp/crsp_msf_*.csv.gz` | `PERMNO date SHRCD EXCHCD PRC SHROUT RET DLRET SICCD` | WRDS: CRSP monthly stock file, 2012-01 to 2025-12 |
| `interim/universe_links.csv` | `cik`, `PERMNO`, `year` (universe year = July Y to June Y+1), `firm`, `ticker` | CRSP–Compustat CUSIP links to EDGAR CIKs, audited point-in-time |
| `interim/universe_permno_year.csv` | `year`, `PERMNO` | Top 1,500 stocks by June market cap, share codes 10/11, exchanges 1/2/3 |

## Downloading 10-K text

```bash
export SEC_UA="Your Name research project you@example.com"   # SEC fair-access policy
python src/edgar_item1.py --ciks data/interim/universe_ciks.csv \
    --out data/interim/public_text.parquet --start 2013 --end 2025
```

The download is rate-limited to SEC's published 10 requests per second and can be resumed. Every filing is cached, and the extraction method is recorded for each one.

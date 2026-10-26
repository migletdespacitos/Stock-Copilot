from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

import config

log = logging.getLogger(__name__)

def fetch_price_history(ticker, period = config.PRICE_HISTORY_PERIOD):
    """Daily OHLCV, split/dividend adjusted so candles and indicators share one price basis."""
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"No price data returned for '{ticker}'. Check the symbol.")

    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
    dates = pd.to_datetime(hist[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    df = pd.DataFrame(
        {
            "date": dates.dt.normalize(),
            "open": hist["Open"].astype(float),
            "high": hist["High"].astype(float),
            "low": hist["Low"].astype(float),
            "close": hist["Close"].astype(float),
            "volume": hist["Volume"].fillna(0).astype("int64"),
        }
    )
    return (
        df.dropna(subset=["close"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

_REVENUE_ROWS = ("Total Revenue", "Operating Revenue", "Revenue")
_NET_INCOME_ROWS = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income From Continuing Operation Net Minority Interest",
)

def _first_row(stmt, candidates):
    for name in candidates:
        if name in stmt.index:
            return pd.to_numeric(stmt.loc[name], errors="coerce")
    return pd.Series(float("nan"), index=stmt.columns)

def fetch_annual_fundamentals(ticker):
    """Annual revenue & net income (yfinance typically returns the last ~4 fiscal years)."""
    stmt = yf.Ticker(ticker).income_stmt
    if stmt is None or stmt.empty:
        return pd.DataFrame(columns=["period_end", "revenue", "net_income"])

    df = pd.DataFrame(
        {
            "period_end": pd.to_datetime(stmt.columns),
            "revenue": _first_row(stmt, _REVENUE_ROWS).to_numpy(),
            "net_income": _first_row(stmt, _NET_INCOME_ROWS).to_numpy(),
        }
    )
    return (
        df.dropna(subset=["revenue", "net_income"], how="all")
        .sort_values("period_end")
        .reset_index(drop=True)
    )

def fetch_company_profile(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # yfinance .info is the flakiest endpoint; never block the UI on it
        info = {}
    return {
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
    }

SECTION_LABELS = {
    "item_1": "Item 1 — Business",
    "item_1a": "Item 1A — Risk Factors",
    "item_7": "Item 7 — MD&A",
}

@dataclass
class FilingSection:
    ticker: str
    accession: str
    period_of_report: str  # YYYY-MM-DD (fiscal year end)
    filed_date: str        # YYYY-MM-DD
    section: str           # item_1 | item_1a | item_7
    text: str

    @property
    def label(self) -> str:
        return SECTION_LABELS[self.section]


_S = r"[\s\.\:\-\u2013\u2014]*"  # separators between "Item", number and title
_SECTION_RULES = {
    "item_1": (rf"item{_S}1{_S}business", rf"item{_S}1a{_S}risk"),
    "item_1a": (
        rf"item{_S}1a{_S}risk{_S}factors",
        rf"item{_S}1b\b|item{_S}1c\b|item{_S}2{_S}properties",
    ),
    "item_7": (rf"item{_S}7{_S}management", rf"item{_S}7a\b|item{_S}8{_S}financial"),
}
_BLOCK_TAGS = ["p", "div", "tr", "td", "th", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]


def _compile(body, anchored):
    prefix = r"^[ \t]*" if anchored else r"\b"
    return re.compile(prefix + rf"(?:{body})", re.IGNORECASE | re.MULTILINE)

def _html_to_text(raw):
    """Block-aware HTML → text. Headings land at line starts; inline cross-references don't."""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_after("\n")

    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _span(text, start_re, end_re):
    """Longest 'tight' start→end span. Table-of-contents hits give short spans and lose."""
    starts = [m.start() for m in start_re.finditer(text)]
    ends = [m.start() for m in end_re.finditer(text)]
    best = ""
    for i, s in enumerate(starts):
        e = next((x for x in ends if x > s), None)
        if e is None:
            continue
        if i + 1 < len(starts) and starts[i + 1] < e:
            continue  # a later heading sits closer to this end marker
        if e - s > len(best):
            best = text[s:e]
    return best.strip()


def extract_section(text, key, min_chars = 2000):
    start_body, end_body = _SECTION_RULES[key]
    for anchored in (True, False):
        found = _span(text, _compile(start_body, anchored), _compile(end_body, anchored))
        if len(found) >= min_chars:
            return found
    return ""


def _parse_header(full_submission):
    meta = {"period_of_report": "", "filed_date": ""}
    if not full_submission.exists():
        return meta
    with full_submission.open("r", encoding="utf-8", errors="ignore") as fh:
        head = fh.read(30_000)
    for key, pattern in (
        ("period_of_report", r"CONFORMED PERIOD OF REPORT:\s*(\d{8})"),
        ("filed_date", r"FILED AS OF DATE:\s*(\d{8})"),
    ):
        m = re.search(pattern, head)
        if m:
            d = m.group(1)
            meta[key] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return meta


def _load_primary_text(folder):
    primary = sorted(folder.glob("primary-document.*"))
    if primary:
        return _html_to_text(primary[0].read_text(encoding="utf-8", errors="ignore"))
    raw = (folder / "full-submission.txt").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>", raw, re.S | re.I)
    return _html_to_text(m.group(1) if m else raw)


def download_10k_filings(ticker, limit = 1):
    t = ticker.upper()
    dl = Downloader(config.SEC_COMPANY_NAME, config.SEC_EMAIL, str(config.SEC_DIR))
    dl.get("10-K", t, limit=limit, download_details=True)
    root = config.SEC_DIR / "sec-edgar-filings" / t / "10-K"
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def fetch_10k_sections(ticker, limit = 1):
    """Download the latest `limit` 10-Ks and extract Items 1, 1A and 7."""
    t = ticker.upper()
    folders = download_10k_filings(t, limit)
    if not folders:
        raise ValueError(f"No 10-K found on EDGAR for {t} (foreign issuers file 20-F instead).")

    filings = [(f, _parse_header(f / "full-submission.txt")) for f in folders]
    filings.sort(key=lambda fm: fm[1]["filed_date"], reverse=True)

    sections: list[FilingSection] = []
    for folder, meta in filings[:limit]:
        text = _load_primary_text(folder)
        for key in SECTION_LABELS:
            body = extract_section(text, key)
            if body:
                sections.append(
                    FilingSection(t, folder.name, meta["period_of_report"], meta["filed_date"], key, body)
                )
            else:
                log.warning("Could not locate %s in %s %s", key, t, folder.name)

    if not sections:
        raise ValueError(f"Downloaded {t} 10-K but could not locate Items 1 / 1A / 7.")
    return sections
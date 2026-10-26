from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path

import duckdb
import pandas as pd

import config

_DDL = """
CREATE TABLE IF NOT EXISTS prices (
    ticker  VARCHAR NOT NULL,
    date    DATE    NOT NULL,
    open    DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume  BIGINT,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker      VARCHAR NOT NULL,
    period_end  DATE    NOT NULL,
    revenue     DOUBLE,
    net_income  DOUBLE,
    PRIMARY KEY (ticker, period_end)
);
"""

_TECHNICALS_VIEW = """
CREATE OR REPLACE VIEW v_technicals AS
WITH base AS (
    SELECT ticker, date, open, high, low, close, volume,
           close - LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS delta,
           ROW_NUMBER()       OVER (PARTITION BY ticker ORDER BY date) AS rn
    FROM prices
),
win AS (
    SELECT *,
        AVG(close)        OVER w20  AS sma_20_raw,
        STDDEV_POP(close) OVER w20  AS std_20_raw,
        AVG(close)        OVER w50  AS sma_50_raw,
        AVG(close)        OVER w200 AS sma_200_raw,
        AVG(CASE WHEN delta > 0 THEN delta  ELSE 0 END) OVER w14 AS avg_gain,
        AVG(CASE WHEN delta < 0 THEN -delta ELSE 0 END) OVER w14 AS avg_loss,
        MAX(high) OVER w252 AS high_52w,
        MIN(low)  OVER w252 AS low_52w
    FROM base
    WINDOW
        w14  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13  PRECEDING AND CURRENT ROW),
        w20  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19  PRECEDING AND CURRENT ROW),
        w50  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49  PRECEDING AND CURRENT ROW),
        w200 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),
        w252 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
),
ind AS (
    SELECT ticker, date, open, high, low, close, volume, high_52w, low_52w,
        CASE WHEN rn >= 50  THEN sma_50_raw  END AS sma_50,
        CASE WHEN rn >= 200 THEN sma_200_raw END AS sma_200,
        CASE WHEN rn >= 20  THEN sma_20_raw  END AS bb_mid,
        CASE WHEN rn >= 20  THEN sma_20_raw + 2 * std_20_raw END AS bb_upper,
        CASE WHEN rn >= 20  THEN sma_20_raw - 2 * std_20_raw END AS bb_lower,
        CASE WHEN rn > 14 THEN
            CASE WHEN avg_loss = 0 THEN 100.0
                 ELSE 100 - 100 / (1 + avg_gain / avg_loss) END
        END AS rsi_14
    FROM win
)
SELECT *,
    close - LAG(close) OVER p                     AS change,
    close / NULLIF(LAG(close) OVER p, 0) - 1      AS change_pct,
    close / NULLIF(LAG(close, 21) OVER p, 0) - 1  AS ret_1m,
    close / NULLIF(LAG(close, 63) OVER p, 0) - 1  AS ret_3m,
    (bb_upper - bb_lower) / NULLIF(bb_mid, 0)       AS bb_bandwidth,
    (close - bb_lower) / NULLIF(bb_upper - bb_lower, 0) AS bb_pct_b,
    CASE
        WHEN sma_50 > sma_200 AND LAG(sma_50) OVER p <= LAG(sma_200) OVER p THEN 'GOLDEN_CROSS'
        WHEN sma_50 < sma_200 AND LAG(sma_50) OVER p >= LAG(sma_200) OVER p THEN 'DEATH_CROSS'
    END AS cross_signal
FROM ind
WINDOW p AS (PARTITION BY ticker ORDER BY date);
"""

_FUNDAMENTALS_VIEW = """
CREATE OR REPLACE VIEW v_fundamentals AS
SELECT ticker, period_end, YEAR(period_end) AS fiscal_year, revenue, net_income,
    net_income / NULLIF(revenue, 0) * 100 AS net_margin_pct,
    (revenue - LAG(revenue) OVER f) / NULLIF(ABS(LAG(revenue) OVER f), 0) * 100
        AS revenue_yoy_pct,
    (net_income - LAG(net_income) OVER f) / NULLIF(ABS(LAG(net_income) OVER f), 0) * 100
        AS net_income_yoy_pct
FROM fundamentals
WINDOW f AS (PARTITION BY ticker ORDER BY period_end);
"""

def _row_to_dict(df):
    if df.empty:
        return {}
    rec = df.iloc[0].to_dict()
    return {k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in rec.items()}

class SQLEngine:
    """Thread-safe wrapper around one DuckDB connection (Streamlit serves sessions on threads)."""

    def __init__(self, db_path = config.DUCKDB_PATH):
        self._con = duckdb.connect(str(db_path))
        self._lock = threading.Lock()
        with self._lock:
            for stmt in (_DDL, _TECHNICALS_VIEW, _FUNDAMENTALS_VIEW):
                self._con.execute(stmt)

    def _query(self, sql, params = None):
        with self._lock:
            return self._con.execute(sql, params or []).fetchdf()

    def _replace(self, table, ticker, frame, insert_sql):
        with self._lock:
            self._con.register("_incoming", frame)
            try:
                self._con.execute("BEGIN TRANSACTION")
                self._con.execute(f"DELETE FROM {table} WHERE ticker = ?", [ticker])
                self._con.execute(insert_sql)
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise
            finally:
                self._con.unregister("_incoming")
        return len(frame)

    # ── writes ──
    def upsert_prices(self, ticker, df):
        t = ticker.upper()
        frame = df.assign(ticker=t)[["ticker", "date", "open", "high", "low", "close", "volume"]]
        return self._replace(
            "prices", t, frame,
            "INSERT INTO prices SELECT ticker, CAST(date AS DATE), open, high, low, close, "
            "CAST(volume AS BIGINT) FROM _incoming",
        )

    def upsert_fundamentals(self, ticker, df):
        if df.empty:
            return 0
        t = ticker.upper()
        frame = df.assign(ticker=t)[["ticker", "period_end", "revenue", "net_income"]]
        return self._replace(
            "fundamentals", t, frame,
            "INSERT INTO fundamentals SELECT ticker, CAST(period_end AS DATE), revenue, net_income "
            "FROM _incoming",
        )

    # ── reads ──
    def price_stats(self, ticker):
        row = _row_to_dict(self._query(
            "SELECT COUNT(*) AS n_bars, MIN(date) AS first_date, MAX(date) AS last_date "
            "FROM prices WHERE ticker = ?", [ticker.upper()]))
        row["n_bars"] = int(row.get("n_bars") or 0)
        return row

    def get_technicals(self, ticker: str, start = None):
        sql = "SELECT * FROM v_technicals WHERE ticker = ?"
        params: list = [ticker.upper()]
        if start is not None:
            sql += " AND date >= ?"
            params.append(start)
        return self._query(sql + " ORDER BY date", params)

    def technical_snapshot(self, ticker):
        return _row_to_dict(self._query(
            """
            SELECT *,
                CASE WHEN rsi_14 IS NULL THEN 'N/A'
                     WHEN rsi_14 >= 70 THEN 'OVERBOUGHT'
                     WHEN rsi_14 <= 30 THEN 'OVERSOLD'
                     ELSE 'NEUTRAL' END AS rsi_state,
                CASE WHEN sma_50 IS NULL OR sma_200 IS NULL THEN 'N/A'
                     WHEN sma_50 > sma_200 THEN 'BULLISH'
                     ELSE 'BEARISH' END AS sma_trend
            FROM v_technicals WHERE ticker = ?
            ORDER BY date DESC LIMIT 1
            """, [ticker.upper()]))

    def last_cross(self, ticker):
        row = _row_to_dict(self._query(
            "SELECT date, cross_signal FROM v_technicals "
            "WHERE ticker = ? AND cross_signal IS NOT NULL ORDER BY date DESC LIMIT 1",
            [ticker.upper()]))
        return row or None

    def get_fundamentals(self, ticker):
        return self._query(
            "SELECT * FROM v_fundamentals WHERE ticker = ? ORDER BY period_end", [ticker.upper()])


def classify_momentum(snap, last_cross = None):
    """Collapse the SQL indicators into a transparent, rule-based momentum bias."""

    def ok(x):
        return x is not None and not pd.isna(x)

    score, drivers = 0.0, []

    if snap.get("sma_trend") == "BULLISH":
        score += 1
        drivers.append("SMA50 above SMA200 (long-term uptrend)")
    elif snap.get("sma_trend") == "BEARISH":
        score -= 1
        drivers.append("SMA50 below SMA200 (long-term downtrend)")

    close, sma200 = snap.get("close"), snap.get("sma_200")
    if ok(close) and ok(sma200):
        above = close > sma200
        score += 1 if above else -1
        drivers.append(f"Price {'above' if above else 'below'} SMA200 ({close / sma200 - 1:+.1%})")

    r3 = snap.get("ret_3m")
    if ok(r3):
        score += 1 if r3 > 0 else -1
        drivers.append(f"3-month return {r3:+.1%}")

    rsi = snap.get("rsi_14")
    if ok(rsi):
        if rsi >= 70:
            score += 0.5
            drivers.append(f"RSI {rsi:.0f}: overbought, momentum stretched")
        elif rsi <= 30:
            score -= 0.5
            drivers.append(f"RSI {rsi:.0f}: oversold, selling pressure heavy")
        else:
            score += 0.5 if rsi > 50 else -0.5
            drivers.append(f"RSI {rsi:.0f}: {'above' if rsi > 50 else 'below'} the 50 midline")

    if last_cross and ok(last_cross.get("date")) and ok(snap.get("date")):
        days = (pd.Timestamp(snap["date"]) - pd.Timestamp(last_cross["date"])).days
        if days <= 45:
            golden = last_cross["cross_signal"] == "GOLDEN_CROSS"
            score += 0.5 if golden else -0.5
            drivers.append(f"{'Golden' if golden else 'Death'} cross {days} days ago")

    bias = "BULLISH" if score >= 1.5 else "BEARISH" if score <= -1.5 else "NEUTRAL"
    return {"bias": bias, "score": round(score, 1), "drivers": drivers}

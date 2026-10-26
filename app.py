from __future__ import annotations

import datetime as dt
import html
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from data_ingestion import (fetch_10k_sections, fetch_annual_fundamentals, fetch_company_profile,
                            fetch_price_history)
from rag_engine import OllamaClient, OllamaError, RAGEngine, RetrievedChunk
from sql_engine import SQLEngine, classify_momentum

st.set_page_config(page_title="Retail Stock Copilot", page_icon="📟", layout="wide",
                   initial_sidebar_state="expanded")

C = config.COLORS
LOOKBACK_MONTHS = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60}
VIEWS = ["FILING TL;DR", "FUNDAMENTALS", "10-K Q&A"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, .stApp, .stApp * {{
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}}
.stApp [data-testid="stIconMaterial"], .stApp [class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded' !important;   /* keep Streamlit's icon glyphs */
}}
*, *::before, *::after {{ border-radius: 0 !important; }}

.stApp {{ background: {C['bg']}; color: {C['text']}; }}
[data-testid="stHeader"] {{ background: {C['bg']}; }}
[data-testid="stSidebar"] {{ background: {C['panel']}; border-right: 1px solid {C['grid']}; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 7rem; max-width: 100%; }}
h1, h2, h3, h4, h5 {{ color: {C['amber']} !important; letter-spacing: .05em; text-transform: uppercase; }}
hr {{ border-color: {C['grid']} !important; }}

.stButton > button {{
    width: 100%; background: {C['panel']}; color: {C['amber']}; border: 1px solid {C['amber']};
    text-transform: uppercase; letter-spacing: .08em; font-weight: 700;
}}
.stButton > button:hover {{ background: {C['amber']}; color: {C['bg']}; border-color: {C['amber']}; }}
.stButton > button:disabled {{ color: {C['muted']}; border-color: {C['grid']}; background: {C['panel']}; }}

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
    background: {C['bg']} !important; border-color: {C['grid']} !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
    border-color: {C['amber']} !important;
}}
input, textarea {{ color: {C['text']} !important; caret-color: {C['amber']}; }}
[data-testid="stChatInput"] {{ background: {C['panel']}; border: 1px solid {C['amber']}; }}
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: {C['bg']}; }}
[data-testid="stChatMessage"] {{ background: {C['panel']}; border: 1px solid {C['grid']}; }}
[data-testid="stExpander"] details {{ background: {C['panel']}; border: 1px solid {C['grid']}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {C['grid']}; }}
[data-testid="stButtonGroup"] button {{ background: {C['panel']}; border: 1px solid {C['grid']}; color: {C['muted']}; }}
[data-testid="stButtonGroup"] button[kind$="Active"] {{ background: {C['amber']}; color: {C['bg']}; border-color: {C['amber']}; }}

.terminal-header {{
    display: flex; overflow-x: auto; white-space: nowrap; background: {C['panel']};
    border-top: 2px solid {C['amber']}; border-bottom: 1px solid {C['grid']};
    padding: .45rem 0; margin-bottom: .8rem; font-size: .8rem;
}}
.terminal-header .cell {{ padding: 0 .85rem; border-right: 1px solid {C['grid']}; }}
.terminal-header .k {{ color: {C['muted']}; margin-right: .4rem; }}
.terminal-header .brand {{ background: {C['amber']}; color: {C['bg']}; font-weight: 700; padding: 0 .7rem; }}
.pos {{ color: {C['green']} !important; }}
.neg {{ color: {C['red']} !important; }}
.amb {{ color: {C['amber']} !important; }}
.mut {{ color: {C['muted']} !important; }}

.badge-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: .6rem; margin-bottom: .8rem; }}
.badge {{ background: {C['panel']}; border: 1px solid {C['grid']}; border-left: 3px solid {C['amber']}; padding: .6rem .8rem; }}
.badge.b-pos {{ border-left-color: {C['green']}; }}
.badge.b-neg {{ border-left-color: {C['red']}; }}
.badge .lbl {{ color: {C['muted']}; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; }}
.badge .val {{ font-size: 1.3rem; font-weight: 700; margin: .15rem 0; }}
.badge .sub {{ color: {C['muted']}; font-size: .7rem; }}

.alert {{ background: {C['panel']}; border: 1px solid {C['grid']}; border-left: 4px solid {C['amber']}; padding: .8rem 1rem; }}
.alert.ALIGNED {{ border-left-color: {C['green']}; }}
.alert.DIVERGENT {{ border-left-color: {C['red']}; }}
.alert .title {{ color: {C['amber']}; font-weight: 700; letter-spacing: .1em; font-size: .78rem; }}
.alert .headline {{ font-size: 1.02rem; font-weight: 700; margin: .45rem 0 .3rem; color: #E6EDF3; }}
.alert .meta {{ color: {C['muted']}; font-size: .72rem; margin-bottom: .45rem; }}
.alert .body {{ font-size: .84rem; line-height: 1.55; }}
.tag {{ display: inline-block; padding: .05rem .5rem; font-weight: 700; font-size: .72rem; letter-spacing: .1em; margin-right: .6rem; }}
.tag.ALIGNED {{ background: {C['green']}; color: {C['bg']}; }}
.tag.DIVERGENT {{ background: {C['red']}; color: #fff; }}
.tag.MIXED, .tag.PENDING {{ background: {C['amber']}; color: {C['bg']}; }}

.panel {{ background: {C['panel']}; border: 1px solid {C['grid']}; padding: .8rem 1rem; margin-bottom: .8rem; }}
.panel-title {{ color: {C['amber']}; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; font-size: .78rem;
               border-bottom: 1px solid {C['grid']}; padding-bottom: .35rem; margin-bottom: .55rem; }}
.panel ul {{ margin: 0; padding-left: 1.1rem; }}
.panel li {{ margin-bottom: .35rem; font-size: .84rem; line-height: 1.5; }}
table.term {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
table.term th {{ color: {C['amber']}; text-align: right; border-bottom: 1px solid {C['amber']}; padding: .35rem .5rem; font-weight: 500; }}
table.term td {{ text-align: right; padding: .32rem .5rem; border-bottom: 1px solid {C['grid']}; }}
table.term th:first-child, table.term td:first-child {{ text-align: left; }}
.cite {{ color: {C['amber']}; font-size: .74rem; font-weight: 700; }}
.src {{ border-left: 2px solid {C['grid']}; padding-left: .6rem; margin: .55rem 0; font-size: .76rem; }}
.src .head {{ color: {C['muted']}; margin-bottom: .2rem; }}
.sys div {{ font-size: .76rem; margin: .15rem 0; }}
.footer-note {{ color: {C['muted']}; font-size: .68rem; margin-top: 1.2rem; border-top: 1px solid {C['grid']}; padding-top: .5rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

esc = html.escape


def _num(x):
    return x is not None and not pd.isna(x)


def fmt_num(x, d = 2):
    return f"{x:,.{d}f}" if _num(x) else "—"


def fmt_pct(x, d = 1, scale = 1.0):
    return f"{x * scale:+.{d}f}%" if _num(x) else "—"


def fmt_compact(x, prefix = ""):
    if not _num(x):
        return "—"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{'-' if x < 0 else ''}{prefix}{abs(x) / div:,.2f}{suf}"
    return f"{prefix}{x:,.0f}"


def sign_cls(x):
    if not _num(x):
        return "mut"
    return "pos" if x > 0 else "neg" if x < 0 else "mut"


def cite_html(text):
    return re.sub(r"\[S(\d+)\]", r'<span class="cite">[S\1]</span>', esc(text or ""))


BIAS_CLS = {"BULLISH": "pos", "BEARISH": "neg", "NEUTRAL": "amb", "N/A": "mut"}

@st.cache_resource(show_spinner="Booting DuckDB analytical engine…")
def get_sql():
    return SQLEngine()


@st.cache_resource(show_spinner="Loading bge-small-en-v1.5 + local Qdrant (first run downloads ~130 MB)…")
def get_rag():
    return RAGEngine()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_profile(ticker):
    return fetch_company_profile(ticker)


@st.cache_data(ttl=20, show_spinner=False)
def ollama_status():
    probe = OllamaClient()
    up = probe.is_up()
    return up, (probe.list_models() if up else [])


ss = st.session_state
ss.setdefault("analysis", {})
ss.setdefault("chat", {})

def run_ingestion(ticker, n_filings, sql, rag):
    with st.status(f"INGESTING {ticker} …", expanded=True) as status:
        try:
            status.write("› yfinance · daily OHLCV (5Y, adjusted)")
            n = sql.upsert_prices(ticker, fetch_price_history(ticker))
            status.write(f"  ✓ {n:,} bars → DuckDB `prices`")
        except Exception as exc:
            status.update(label=f"PRICE INGESTION FAILED · {exc}", state="error")
            return False

        try:
            status.write("› yfinance · annual income statement")
            n = sql.upsert_fundamentals(ticker, fetch_annual_fundamentals(ticker))
            status.write(f"  ✓ {n} fiscal years → DuckDB `fundamentals`")
        except Exception as exc:
            status.write(f"  ⚠ fundamentals unavailable: {exc}")

        try:
            status.write("› SEC EDGAR · 10-K (Items 1, 1A, 7)")
            sections = fetch_10k_sections(ticker, limit=n_filings)
            status.write(f"  ✓ {len(sections)} sections parsed")
            status.write("› bge-small-en-v1.5 → Qdrant")
            status.write(f"  ✓ {rag.index_sections(sections):,} chunks indexed")
        except Exception as exc:
            status.write(f"  ⚠ filing ingestion failed: {exc}")

        ss.analysis.pop(ticker, None)
        status.update(label=f"{ticker} READY", state="complete", expanded=False)
    return True

def render_header(ticker, profile, snap, meta):
    now = dt.datetime.now().astimezone()
    chg, chg_pct = snap.get("change"), snap.get("change_pct")
    as_of = pd.Timestamp(snap["date"]).strftime("%Y-%m-%d") if snap.get("date") is not None else "—"
    cells = [
        '<span class="cell brand">RSC ▮ TERMINAL</span>',
        f'<span class="cell amb"><b>{esc(ticker)} US</b></span>',
        f'<span class="cell mut">{esc(profile.get("name", ""))[:40]}</span>',
        f'<span class="cell"><span class="k">LAST</span>{fmt_num(snap.get("close"))}</span>',
        f'<span class="cell {sign_cls(chg)}"><span class="k">CHG</span>{fmt_num(chg) if not _num(chg) else f"{chg:+,.2f}"}</span>',
        f'<span class="cell {sign_cls(chg_pct)}"><span class="k">CHG%</span>{fmt_pct(chg_pct, 2, 100)}</span>',
        f'<span class="cell"><span class="k">VOL</span>{fmt_compact(snap.get("volume"))}</span>',
        f'<span class="cell"><span class="k">52W H</span>{fmt_num(snap.get("high_52w"))}</span>',
        f'<span class="cell"><span class="k">52W L</span>{fmt_num(snap.get("low_52w"))}</span>',
        f'<span class="cell"><span class="k">MKT CAP</span>{fmt_compact(profile.get("market_cap"), "$")}</span>',
        f'<span class="cell"><span class="k">10-K</span>{esc(meta.get("period_of_report") or "NOT INDEXED")}</span>',
        f'<span class="cell"><span class="k">AS OF</span>{as_of}</span>',
        f'<span class="cell mut">{now:%H:%M %Z}</span>',
    ]
    st.markdown(f'<div class="terminal-header">{"".join(cells)}</div>', unsafe_allow_html=True)


def _badge(label, value, sub = "", cls = "amb"):
    border = {"pos": "b-pos", "neg": "b-neg"}.get(cls, "")
    return (f'<div class="badge {border}"><div class="lbl">{esc(label)}</div>'
            f'<div class="val {cls}">{esc(value)}</div><div class="sub">{sub}</div></div>')


def render_badges(snap, momentum, last_cross, fund):
    rsi, rsi_state = snap.get("rsi_14"), snap.get("rsi_state", "N/A")
    rsi_cls = {"OVERBOUGHT": "neg", "OVERSOLD": "pos", "NEUTRAL": "amb"}.get(rsi_state, "mut")

    trend = snap.get("sma_trend", "N/A")
    cross_txt = ""
    if last_cross:
        kind = "GOLDEN" if last_cross["cross_signal"] == "GOLDEN_CROSS" else "DEATH"
        cross_txt = f" · LAST {kind} X {pd.Timestamp(last_cross['date']):%Y-%m-%d}"

    latest = fund.iloc[-1] if not fund.empty else None
    rev_yoy = latest["revenue_yoy_pct"] if latest is not None else None
    ni_yoy = latest["net_income_yoy_pct"] if latest is not None else None
    fy = f"FY{int(latest['fiscal_year'])}" if latest is not None else "N/A"

    badges = [
        _badge("RSI (14)", fmt_num(rsi, 1), esc(rsi_state), rsi_cls),
        _badge("SMA TREND 50/200", trend,
               f"{fmt_num(snap.get('sma_50'))} / {fmt_num(snap.get('sma_200'))}{esc(cross_txt)}",
               BIAS_CLS.get(trend, "mut")),
        _badge("BOLLINGER %B", fmt_num(snap.get("bb_pct_b"), 2),
               f"BANDWIDTH {fmt_pct(snap.get('bb_bandwidth'), 1, 100).lstrip('+')}",
               "neg" if _num(snap.get("bb_pct_b")) and snap["bb_pct_b"] > 1
               else "pos" if _num(snap.get("bb_pct_b")) and snap["bb_pct_b"] < 0 else "amb"),
        _badge("YOY REVENUE", fmt_pct(rev_yoy),
               f"{fy} · {fmt_compact(latest['revenue'] if latest is not None else None, '$')}", sign_cls(rev_yoy)),
        _badge("YOY NET INCOME", fmt_pct(ni_yoy),
               f"{fy} · {fmt_compact(latest['net_income'] if latest is not None else None, '$')}", sign_cls(ni_yoy)),
        _badge("MOMENTUM BIAS", momentum["bias"], f"RULE SCORE {momentum['score']:+.1f}",
               BIAS_CLS[momentum["bias"]]),
    ]
    st.markdown(f'<div class="badge-grid">{"".join(badges)}</div>', unsafe_allow_html=True)


def render_alert(momentum, result, n_chunks):
    if result:
        al = result["alignment"]
        conf = f" · CONF {result['tone_confidence']:.0%}" if result.get("tone_confidence") is not None else ""
        body = f"""
        <div class="alert {al}">
          <span class="tag {al}">{al}</span><span class="title">QUANT-FUNDAMENTAL DIVERGENCE ALERT</span>
          <div class="headline">{cite_html(result['headline'])}</div>
          <div class="meta">TECHNICALS <b class="{BIAS_CLS[result['technical_bias']]}">{result['technical_bias']}</b>
            ({result['technical_score']:+.1f}) · 10-K TONE <b class="{BIAS_CLS[result['filing_tone']]}">{result['filing_tone']}</b>{conf}
            · {esc(result['model'])} · {esc(result['generated_at'])}</div>
          <div class="body">{cite_html(result['explanation'])}</div>
        </div>"""
    else:
        drivers = " · ".join(esc(d) for d in momentum["drivers"]) or "insufficient history"
        filing = f"{n_chunks:,} 10-K chunks indexed" if n_chunks else "no 10-K indexed yet"
        body = f"""
        <div class="alert">
          <span class="tag PENDING">PENDING</span><span class="title">QUANT-FUNDAMENTAL DIVERGENCE ALERT</span>
          <div class="headline">Technical bias is <span class="{BIAS_CLS[momentum['bias']]}">{momentum['bias']}</span>.
            Run the synthesis to test it against the 10-K.</div>
          <div class="meta">{filing}</div>
          <div class="body">{drivers}</div>
        </div>"""
    st.markdown(body, unsafe_allow_html=True)


def build_price_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.74, 0.26])
    x = df["date"]
    line = lambda color, width=1.0, dash=None: dict(color=color, width=width, dash=dash)  # noqa: E731

    fig.add_trace(go.Scatter(x=x, y=df["bb_upper"], name="BB UPPER", line=line(C["muted"], 1, "dot"),
                             hovertemplate="%{y:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["bb_lower"], name="BB LOWER", line=line(C["muted"], 1, "dot"),
                             fill="tonexty", fillcolor="rgba(255,158,0,0.05)", hovertemplate="%{y:.2f}"),
                  row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC",
        increasing=dict(line=dict(color=C["green"], width=1), fillcolor=C["green"]),
        decreasing=dict(line=dict(color=C["red"], width=1), fillcolor=C["red"]),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["bb_mid"], name="SMA 20", line=line("#3A4660", 1),
                             hovertemplate="%{y:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["sma_50"], name="SMA 50", line=line(C["amber"], 1.6),
                             hovertemplate="%{y:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["sma_200"], name="SMA 200", line=line(C["cyan"], 1.6),
                             hovertemplate="%{y:.2f}"), row=1, col=1)

    crosses = df[df["cross_signal"].notna()]
    for sig, symbol, color in (("GOLDEN_CROSS", "triangle-up", C["green"]),
                               ("DEATH_CROSS", "triangle-down", C["red"])):
        sub = crosses[crosses["cross_signal"] == sig]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["sma_50"], mode="markers", name=sig.replace("_", " "),
                marker=dict(symbol=symbol, size=13, color=color, line=dict(color=C["bg"], width=1)),
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["rsi_14"], name="RSI 14", line=line(C["amber"], 1.4),
                             hovertemplate="%{y:.1f}"), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,158,0,0.04)", line_width=0, row=2, col=1)
    fig.add_hline(y=70, line=line(C["red"], 1, "dot"), row=2, col=1)
    fig.add_hline(y=30, line=line(C["green"], 1, "dot"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], height=680,
        margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
        font=dict(family="JetBrains Mono, monospace", color=C["text"], size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor=C["bg"], bordercolor=C["amber"], font=dict(family="JetBrains Mono, monospace")),
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False, showline=True, linecolor=C["grid"],
                     rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])],
                     showspikes=True, spikecolor=C["amber"], spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, showline=True, linecolor=C["grid"], side="right")
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=2, col=1)
    return fig


def render_fundamentals(fund):
    if fund.empty:
        st.markdown('<div class="panel mut">No annual financials available from yfinance for this ticker.</div>',
                    unsafe_allow_html=True)
        return
    rows = "".join(
        f"<tr><td>FY{int(r.fiscal_year)} <span class='mut'>({pd.Timestamp(r.period_end):%Y-%m-%d})</span></td>"
        f"<td>{fmt_compact(r.revenue, '$')}</td><td class='{sign_cls(r.revenue_yoy_pct)}'>{fmt_pct(r.revenue_yoy_pct)}</td>"
        f"<td class='{sign_cls(r.net_income)}'>{fmt_compact(r.net_income, '$')}</td>"
        f"<td class='{sign_cls(r.net_income_yoy_pct)}'>{fmt_pct(r.net_income_yoy_pct)}</td>"
        f"<td>{fmt_num(r.net_margin_pct, 1)}%</td></tr>"
        for r in fund.iloc[::-1].itertuples()
    )
    st.markdown(
        f"""<div class="panel"><div class="panel-title">ANNUAL FUNDAMENTALS · DuckDB v_fundamentals (LAG window)</div>
        <table class="term"><tr><th>FISCAL YEAR</th><th>REVENUE</th><th>YoY</th><th>NET INCOME</th><th>YoY</th>
        <th>NET MARGIN</th></tr>{rows}</table></div>""",
        unsafe_allow_html=True,
    )
    labels = [f"FY{int(y)}" for y in fund["fiscal_year"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=fund["revenue"], name="REVENUE", marker_color=C["amber"])
    fig.add_bar(x=labels, y=fund["net_income"], name="NET INCOME",
                marker_color=[C["green"] if v >= 0 else C["red"] for v in fund["net_income"].fillna(0)])
    fig.update_layout(template="plotly_dark", paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], height=300,
                      barmode="group", margin=dict(l=8, r=8, t=30, b=8),
                      font=dict(family="JetBrains Mono, monospace", color=C["text"], size=11),
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["grid"], side="right")
    st.plotly_chart(fig, config={"displaylogo": False})


def render_sources(chunks):
    if not chunks:
        return
    with st.expander(f"SOURCES · {len(chunks)} 10-K EXCERPTS"):
        for i, ch in enumerate(chunks, 1):
            snippet = ch.text if len(ch.text) <= 650 else ch.text[:650] + " …"
            st.markdown(
                f'<div class="src"><div class="head"><span class="cite">[S{i}]</span> {esc(ch.citation)}'
                f' · sim {ch.score:.2f} · filed {esc(ch.filed_date or "n/a")}</div>{esc(snippet)}</div>',
                unsafe_allow_html=True,
            )


def _bullets(title, items):
    lis = "".join(f"<li>{cite_html(i)}</li>" for i in items) or "<li class='mut'>—</li>"
    return f'<div class="panel"><div class="panel-title">{esc(title)}</div><ul>{lis}</ul></div>'


def render_tldr(result):
    if not result:
        st.markdown('<div class="panel mut">Run the synthesis above to generate the plain-English filing summary.</div>',
                    unsafe_allow_html=True)
        return
    left, right = st.columns([3, 2])
    with left:
        st.markdown(_bullets("TL;DR · WHAT THE 10-K SAYS", result["tldr"]), unsafe_allow_html=True)
        st.markdown(_bullets("TECHNICAL DRIVERS (SQL)", result["technical_drivers"]), unsafe_allow_html=True)
    with right:
        st.markdown(_bullets("KEY RISKS", result["key_risks"]), unsafe_allow_html=True)
        st.markdown(_bullets("WATCH LIST", result["watch_items"]), unsafe_allow_html=True)
    render_sources(result["sources"])


def render_chat(ticker, rag, llm, can_chat, reason):
    history = ss.chat.setdefault(ticker, [])

    panel = st.container()  # wrapping the input in a container renders it inline, not pinned
    with panel:
        st.markdown(f'<div class="panel-title">10-K Q&A · {esc(ticker)} · {esc(llm.model)}</div>',
                    unsafe_allow_html=True)
        chat_box = st.container(height=520, border=True)
        prompt = st.chat_input(f"ASK THE {ticker} 10-K › e.g. What are the biggest supply-chain risks?",
                               disabled=not can_chat)

    with chat_box:
        if not history and not prompt:
            st.markdown(f'<div class="mut">Ask anything about the {esc(ticker)} 10-K. Answers cite the '
                        f'exact filing excerpts used. {esc(reason)}</div>', unsafe_allow_html=True)
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                render_sources(msg.get("sources", []))

        if not prompt:
            return

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown('<span class="mut">› searching 10-K excerpts…</span>', unsafe_allow_html=True)
            chunks: list[RetrievedChunk] = []
            try:
                stream, chunks = rag.answer_question(prompt, ticker, llm, history)
                status.markdown(
                    f'<span class="mut">› {esc(llm.model)} reading {len(chunks)} excerpts… '
                    f'(first reply can take 30–90 s on CPU)</span>', unsafe_allow_html=True)

                def tokens():
                    first = True
                    for tok in stream:
                        if first:
                            status.empty()
                            first = False
                        yield tok

                answer = st.write_stream(tokens())
                answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                if not answer.strip():
                    answer = "⚠ The model returned an empty reply. Try another model in the sidebar."
                    st.warning(answer)
            except Exception as exc:  # surface everything; silent failures are the worst UX here
                status.empty()
                answer = f"⚠ {type(exc).__name__}: {exc}"
                st.error(answer)
            render_sources(chunks)

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": answer, "sources": chunks})

sql, rag = get_sql(), get_rag()
ollama_up, installed_models = ollama_status()

with st.sidebar:
    st.markdown("### ◆ CONTROL")
    ticker = st.text_input("TICKER", value="AAPL", max_chars=10).strip().upper()
    lookback = st.selectbox("CHART LOOKBACK", list(LOOKBACK_MONTHS), index=2)
    n_filings = int(st.number_input("10-K FILINGS TO INDEX", min_value=1, max_value=3, value=1))
    refresh = st.button("⟳ INGEST / REFRESH")
    st.markdown("---")
    if installed_models:
        default_idx = next((i for i, m in enumerate(installed_models) if m.startswith(config.DEFAULT_LLM)), 0)
        model = st.selectbox("OLLAMA MODEL", installed_models, index=default_idx)
    else:
        model = st.text_input("OLLAMA MODEL", value=config.DEFAULT_LLM)

if not ticker:
    st.stop()

if refresh or sql.price_stats(ticker)["n_bars"] == 0:
    if not run_ingestion(ticker, n_filings, sql, rag):
        st.stop()

snap = sql.technical_snapshot(ticker)
if not snap:
    st.error(f"No price history stored for {ticker}.")
    st.stop()

last_cross = sql.last_cross(ticker)
momentum = classify_momentum(snap, last_cross)
fund = sql.get_fundamentals(ticker)
start = (pd.Timestamp(snap["date"]) - pd.DateOffset(months=LOOKBACK_MONTHS[lookback])).date()
tech = sql.get_technicals(ticker, start)
profile = get_profile(ticker)
n_chunks = rag.count(ticker)
meta = rag.filing_meta(ticker)
llm = OllamaClient(model=model)
stats = sql.price_stats(ticker)

with st.sidebar:
    st.markdown(
        f"""<div class="panel sys"><div class="panel-title">SYSTEM</div>
        <div>OLLAMA&nbsp;&nbsp;<span class="{'pos' if ollama_up else 'neg'}">● {'ONLINE' if ollama_up else 'OFFLINE'}</span></div>
        <div>DUCKDB&nbsp;&nbsp;<span class="amb">{stats['n_bars']:,}</span> bars</div>
        <div>QDRANT&nbsp;&nbsp;<span class="amb">{n_chunks:,}</span> chunks</div>
        <div>EMBED&nbsp;&nbsp;&nbsp;<span class="mut">bge-small-en-v1.5</span></div></div>""",
        unsafe_allow_html=True,
    )
    if not ollama_up:
        st.caption("Start the LLM with `ollama serve`, then `ollama pull llama3.2`.")

render_header(ticker, profile, snap, meta)
render_badges(snap, momentum, last_cross, fund)

alert_col, action_col = st.columns([6, 1.2], vertical_alignment="center")
with action_col:
    run_synthesis = st.button("▶ RUN SYNTHESIS", disabled=not (ollama_up and n_chunks), key="run_synthesis")
if run_synthesis:
    with alert_col, st.spinner(f"{model} reading the 10-K against the SQL indicators…"):
        try:
            ss.analysis[ticker] = rag.divergence_analysis(ticker, snap, momentum, fund, llm)
        except (OllamaError, ValueError) as exc:
            st.error(f"Synthesis failed: {exc}")
with alert_col:
    render_alert(momentum, ss.analysis.get(ticker), n_chunks)

st.markdown(f'<div class="panel-title" style="margin-top:1rem">{esc(ticker)} · DAILY · SMA 50/200 · '
            f'BOLLINGER (20, 2σ) · RSI (14) · {lookback}</div>', unsafe_allow_html=True)
st.plotly_chart(build_price_chart(tech), config={"displaylogo": False, "scrollZoom": True})

view = st.segmented_control("VIEW", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
if view == "FUNDAMENTALS":
    render_fundamentals(fund)
elif view == "FILING TL;DR":
    render_tldr(ss.analysis.get(ticker))
else:
    reason = "" if ollama_up and n_chunks else (
        "Ollama is offline." if not ollama_up else "No 10-K indexed yet; click INGEST / REFRESH.")
    render_chat(ticker, rag, llm, can_chat=bool(ollama_up and n_chunks), reason=reason)

st.markdown('<div class="footer-note">FOR RESEARCH & EDUCATION ONLY · NOT INVESTMENT ADVICE · '
            'PRICES: YAHOO FINANCE VIA yfinance (UNOFFICIAL) · FILINGS: SEC EDGAR · LLM: LOCAL OLLAMA</div>',
            unsafe_allow_html=True)

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import pandas as pd
import requests
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

import config
from data_ingestion import SECTION_LABELS, FilingSection

@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    ticker: str
    section: str
    period_of_report: str
    filed_date: str
    accession: str
    chunk_index: int

    @property
    def section_label(self):
        return SECTION_LABELS.get(self.section, self.section)

    @property
    def citation(self):
        period = self.period_of_report or "n/a"
        return f"{self.ticker} 10-K · FY ending {period} · {self.section_label} · chunk {self.chunk_index}"


def chunk_words(text, size = config.CHUNK_WORDS, overlap = config.CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if len(window) >= config.MIN_CHUNK_WORDS or not chunks:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks

class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal REST client for a local `ollama serve` (http://localhost:11434)."""

    def __init__(self, model = config.DEFAULT_LLM, base_url = config.OLLAMA_URL, timeout = 600):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_up(self):
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except (requests.RequestException, ValueError, KeyError):
            return []

    def _payload(self, messages, stream, temperature, json_mode):
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            "options": {"temperature": temperature, "num_ctx": config.OLLAMA_NUM_CTX},
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    @staticmethod
    def _check(r: requests.Response):
        if not r.ok:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")

    def chat(self, messages, temperature = 0.1, json_mode = False):
        try:
            r = requests.post(f"{self.base_url}/api/chat",
                              json=self._payload(messages, False, temperature, json_mode),
                              timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        self._check(r)
        return r.json().get("message", {}).get("content", "")

    def stream_chat(self, messages, temperature = 0.2):
        try:
            with requests.post(f"{self.base_url}/api/chat",
                               json=self._payload(messages, True, temperature, False),
                               stream=True, timeout=self.timeout) as r:
                self._check(r)
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise OllamaError(data["error"])
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

_SYSTEM_DIVERGENCE = """You are a buy-side equity analyst writing for retail investors.
You compare a stock's technical momentum (computed from price data) with the tone of its latest 10-K.
Rules:
- Use ONLY the provided filing excerpts for claims about the company and cite them as [S1], [S2], ...
- Judge filing tone from substance (growth, margins, liquidity, severity and specificity of risks),
  not from boilerplate risk language that every 10-K contains.
- Plain English; explain any jargon. Never give buy/sell recommendations.
- Respond with a single JSON object and nothing else."""

_USER_DIVERGENCE = """TICKER: {ticker}

TECHNICAL MOMENTUM (DuckDB SQL indicators)
{technicals}

FUNDAMENTALS (annual income statement)
{fundamentals}

10-K EXCERPTS
{context}

Return JSON with exactly these keys:
{{
  "filing_tone": "BULLISH" | "NEUTRAL" | "BEARISH",
  "tone_confidence": number between 0 and 1,
  "headline": "one sentence, max 25 words, saying whether price momentum and the filing agree",
  "divergence_explanation": "3-5 sentences comparing the technical picture with the filing, with [S#] citations",
  "tldr": ["3-5 plain-English bullets summarising the 10-K, each with a [S#] citation"],
  "key_risks": ["the 3 most material risks in plain English, each with a [S#] citation"],
  "watch_items": ["2-3 specific things a retail investor should monitor next"]
}}"""

_SYSTEM_QA = """You are a 10-K research assistant for retail investors.
Answer strictly from the numbered filing excerpts provided and cite every factual claim as [S#].
If the excerpts do not contain the answer, say so plainly; do not guess or use outside knowledge.
Use plain English and keep answers under ~200 words unless asked for more. Never give buy/sell advice."""

_DIVERGENCE_QUERIES = (
    ("item_7", "revenue growth drivers, demand trends and business outlook"),
    ("item_7", "margin pressure, rising costs, or declining sales"),
    ("item_7", "liquidity, cash flow, debt and capital returns"),
    ("item_1a", "most significant risks that could materially harm the business"),
    ("item_1a", "competition, regulation, supply chain and macroeconomic risks"),
)

_TONES = {"BULLISH", "NEUTRAL", "BEARISH"}

def _parse_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}

def _as_list(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def _alignment(technical_bias, filing_tone):
    """Deterministic — the LLM classifies tone; Python decides alignment."""
    if technical_bias == filing_tone:
        return "ALIGNED"
    if "NEUTRAL" in (technical_bias, filing_tone):
        return "MIXED"
    return "DIVERGENT"


def _jsonable(records):
    out = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                clean[k] = None
            elif isinstance(v, (pd.Timestamp, dt.date)):
                clean[k] = str(pd.Timestamp(v).date())
            elif isinstance(v, float):
                clean[k] = round(v, 4)
            else:
                clean[k] = v
        out.append(clean)
    return out

class RAGEngine:
    def __init__(self, qdrant_path=config.QDRANT_PATH, collection = config.QDRANT_COLLECTION,
                 embed_model = config.EMBED_MODEL):
        self.collection = collection
        self.client = QdrantClient(path=str(qdrant_path))  # embedded, on-disk, no server
        self.embedder = SentenceTransformer(embed_model)
        self._lock = threading.Lock()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
            )

    # ── helpers ──
    @staticmethod
    def _filter(ticker, sections = None):
        must = [models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
        if sections:
            must.append(models.FieldCondition(key="section", match=models.MatchAny(any=list(sections))))
        return models.Filter(must=must)

    def _embed(self, texts):
        return self.embedder.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False).tolist()

    @staticmethod
    def _format_context(chunks):
        return "\n\n".join(
            f"[S{i}] ({c.section_label}, FY ending {c.period_of_report or 'n/a'})\n{c.text}"
            for i, c in enumerate(chunks, 1)
        )

    # ── indexing ──
    def index_sections(self, sections, batch_size = 128):
        if not sections:
            return 0
        ticker = sections[0].ticker.upper()
        records = [(sec, i, chunk) for sec in sections for i, chunk in enumerate(chunk_words(sec.text))]
        vectors = self._embed([r[2] for r in records])  # embed first: a failure leaves the old index intact

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sec.ticker}|{sec.accession}|{sec.section}|{i}")),
                vector=vec,
                payload={
                    "ticker": ticker, "section": sec.section, "accession": sec.accession,
                    "period_of_report": sec.period_of_report, "filed_date": sec.filed_date,
                    "chunk_index": i, "text": chunk,
                },
            )
            for (sec, i, chunk), vec in zip(records, vectors)
        ]
        with self._lock:
            self.client.delete(self.collection,
                               points_selector=models.FilterSelector(filter=self._filter(ticker)))
            for start in range(0, len(points), batch_size):
                self.client.upsert(self.collection, points=points[start:start + batch_size])
        return len(points)

    def count(self, ticker):
        with self._lock:
            return self.client.count(self.collection, count_filter=self._filter(ticker), exact=True).count

    def filing_meta(self, ticker):
        with self._lock:
            points, _ = self.client.scroll(self.collection, scroll_filter=self._filter(ticker),
                                           limit=1, with_payload=True)
        if not points:
            return {}
        p = points[0].payload or {}
        return {k: p.get(k, "") for k in ("period_of_report", "filed_date", "accession")}

    # ── retrieval ──
    def retrieve(self, query, ticker, k = 6,
                 sections = None):
        qv = self.embedder.encode(config.BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        with self._lock:
            res = self.client.query_points(self.collection, query=qv, query_filter=self._filter(ticker, sections),
                                           limit=k, with_payload=True)
        out = []
        for p in res.points:
            pl = p.payload or {}
            out.append(RetrievedChunk(
                id=str(p.id), text=pl.get("text", ""), score=float(p.score), ticker=pl.get("ticker", ""),
                section=pl.get("section", ""), period_of_report=pl.get("period_of_report", ""),
                filed_date=pl.get("filed_date", ""), accession=pl.get("accession", ""),
                chunk_index=int(pl.get("chunk_index", 0)),
            ))
        return out

    # ── synthesis ──
    def divergence_analysis(self, ticker, snapshot, momentum,
                            fundamentals, llm):
        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        for section, query in _DIVERGENCE_QUERIES:
            for ch in self.retrieve(query, ticker, k=2, sections=[section]):
                if ch.id not in seen:
                    seen.add(ch.id)
                    chunks.append(ch)
        if not chunks:
            raise ValueError(f"No indexed 10-K text for {ticker}. Run ingestion first.")

        tech_keys = ("date", "close", "sma_50", "sma_200", "rsi_14", "rsi_state", "sma_trend",
                     "bb_pct_b", "ret_1m", "ret_3m", "high_52w", "low_52w")
        technicals = _jsonable([{k: snapshot.get(k) for k in tech_keys}])[0]
        technicals.update(momentum_bias=momentum["bias"], momentum_score=momentum["score"],
                          drivers=momentum["drivers"])
        fund_cols = ["fiscal_year", "revenue", "net_income", "revenue_yoy_pct", "net_income_yoy_pct", "net_margin_pct"]
        fund_records = _jsonable(fundamentals[fund_cols].to_dict("records")) if not fundamentals.empty else []

        messages = [
            {"role": "system", "content": _SYSTEM_DIVERGENCE},
            {"role": "user", "content": _USER_DIVERGENCE.format(
                ticker=ticker.upper(),
                technicals=json.dumps(technicals, indent=2, default=str),
                fundamentals=json.dumps(fund_records, indent=2, default=str),
                context=self._format_context(chunks),
            )},
        ]
        data = _parse_json(llm.chat(messages, temperature=0.1, json_mode=True))

        tone = str(data.get("filing_tone", "NEUTRAL")).upper().strip()
        tone = tone if tone in _TONES else "NEUTRAL"
        try:
            confidence = min(max(float(data.get("tone_confidence")), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = None

        return {
            "ticker": ticker.upper(),
            "technical_bias": momentum["bias"],
            "technical_score": momentum["score"],
            "technical_drivers": momentum["drivers"],
            "filing_tone": tone,
            "tone_confidence": confidence,
            "alignment": _alignment(momentum["bias"], tone),
            "headline": str(data.get("headline", "")).strip() or "Synthesis returned no headline.",
            "explanation": str(data.get("divergence_explanation", "")).strip(),
            "tldr": _as_list(data.get("tldr")),
            "key_risks": _as_list(data.get("key_risks")),
            "watch_items": _as_list(data.get("watch_items")),
            "sources": chunks,
            "model": llm.model,
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def answer_question(self, question, ticker, llm,
                        history = (), k = 6):
        chunks = self.retrieve(question, ticker, k=k)
        if not chunks:
            return iter(["No indexed 10-K text for this ticker yet. Run ingestion first."]), []

        messages = [{"role": "system", "content": _SYSTEM_QA}]
        for turn in list(history)[-4:]:
            # earlier [S#] labels pointed at different excerpts; strip them to avoid confusion
            messages.append({"role": turn["role"], "content": re.sub(r"\[S\d+\]", "", turn["content"])})
        messages.append({
            "role": "user",
            "content": f"10-K EXCERPTS ({ticker.upper()}):\n{self._format_context(chunks)}\n\nQUESTION: {question}",
        })
        return llm.stream_chat(messages), chunks
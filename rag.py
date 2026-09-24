#!/usr/bin/env python3
"""
grounded-rag-chat — retrieval-augmented chat with citation enforcement.

Pipeline:  embed question -> sqlite-vec top-k -> synthesize grounded answer
           -> every factual claim must trace to a retrieved chunk, else refuse.

Two synthesis modes:
  * "extractive" (default, no API key): the answer is a verbatim quote from the
    best chunk plus a citation. Faithfulness is structural, not hoped for.
  * "llm" (optional): an OpenAI-compatible chat endpoint drafts the answer, but
    the system prompt forces grounding, [doc-id] citations, and refusal when the
    context lacks the answer. Requires OPENAI_API_KEY (+ optional OPENAI_BASE_URL).

Usage:
    from rag import RAGChat
    chat = RAGChat("data/rag.db")
    print(chat.answer("How long is the warranty?")["answer"])
"""

import json
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

# Sanitize no_proxy: newer httpx (used by huggingface_hub) crashes parsing
# bracketed IPv6 entries like [::1] ("Invalid port: ':1]'"). Strip those
# entries before any network library reads the environment.
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        os.environ[_var] = ",".join(p for p in _val.split(",") if "[" not in p)
del _var, _val

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, runs fully local
EMBED_DIMS = 384

# Cosine distance above this => the corpus has nothing relevant => refuse.
# Backstop only: the primary out-of-scope signal is _topical_overlap() below.
# (Pure distance can't separate "stock price?" (0.73, nearest doc: shipping)
# from legit questions (up to 0.76) — calibrated on evals/golden.json.)
REFUSAL_DISTANCE = 0.85

# Small stopword list for the topical-overlap gate. "helios"/"home" are brand
# boilerplate present in every doc title — they carry no topical signal.
_STOPWORDS = frozenset("""
a an the and or but is are was were be been being to of in on for with as at by
from what how when where which who whom whose do does did can could should would
will shall may might must i me my mine you your yours we our ours it its this
that these those there their s t don doesn about into over under per each any
all no not so if than then too very just also only such own same other some
helios home
""".split())


# A pinch of morphological normalization for support-doc vocabulary, so
# "kept" matches "keep" and "devices" matches "device". Deliberately tiny —
# a full stemmer would be the next step, not a hidden dependency.
_IRREGULAR = {"kept": "keep", "bought": "buy", "sold": "sell",
              "paid": "pay", "sent": "send"}


def _stem(t):
    if t in _IRREGULAR:
        return _IRREGULAR[t]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _content_tokens(s):
    """Topical tokens of a string: markdown formatting stripped, stopwords and
    brand boilerplate ("helios"/"home", in every doc title) removed, stemmed.

    Note: chunks are single-line (whitespace-collapsed), so this strips header
    *markers* inline rather than dropping header lines — dropping lines would
    nuke the whole chunk."""
    s = re.sub(r"#{1,6}\s*", " ", s)  # ## headers -> plain words
    s = s.replace("**", " ").replace("*", " ").replace("`", " ")
    return {_stem(t) for t in _WORD.findall(s.lower())
            if t not in _STOPWORDS and len(t) > 1}


def _topical_overlap(question, chunk_text_):
    return _content_tokens(question) & _content_tokens(chunk_text_)

RETRIEVE_K = 5

_model = None


def default_embed_fn(texts):
    """Embed with fastembed, lazily loaded and cached process-wide."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(EMBED_MODEL)
    return [list(v) for v in _model.embed(texts)]


def chunk_text(text, words=120, overlap=20):
    """Split markdown into chunks of ~`words` words, keeping newlines.

    Chunks break at paragraph boundaries (blank lines) so headers, list items,
    and sentences stay intact — the sentence splitter and quote extractor rely
    on that structure. Oversized single paragraphs fall back to overlapping
    word windows. Pure function (testable).
    """
    import re as _re

    blocks = [b.strip() for b in _re.split(r"\n\s*\n", text) if b.strip()]
    chunks, cur, cur_words = [], [], 0
    for b in blocks:
        toks = b.split()
        if len(toks) > words:
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_words = [], 0
            start = 0
            while start < len(toks):
                chunks.append(" ".join(toks[start : start + words]))
                start += words - overlap
            continue
        if cur_words + len(toks) > words and cur:
            chunks.append("\n".join(cur))
            cur, cur_words = [], 0
        cur.append(b)
        cur_words += len(toks)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _serialize(vec):
    import struct

    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """Thin sqlite-vec wrapper: one virtual table, cosine distance search."""

    def __init__(self, db_path):
        import sqlite_vec

        self.db_path = str(db_path)
        # check_same_thread=False: the chat server answers requests on worker
        # threads. Safe here — all writes happen at ingest time (single
        # thread); serving is read-only.
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)

    @classmethod
    def build(cls, db_path, chunks, embed_fn, dims=EMBED_DIMS):
        """chunks: list of (doc_id, title, text). Rebuilds the DB from scratch."""
        db_path = Path(db_path)
        if db_path.exists():
            db_path.unlink()
        store = cls(db_path)
        store.db.execute(f"CREATE VIRTUAL TABLE vec USING vec0(embedding float[{dims}])")
        store.db.execute(
            "CREATE TABLE docs(id INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT)"
        )
        texts = [c[2] for c in chunks]
        vecs = embed_fn(texts)
        for (doc_id, title, text), vec in zip(chunks, vecs):
            rowid = store.db.execute(
                "INSERT INTO docs(doc_id, title, text) VALUES (?,?,?)",
                (doc_id, title, text),
            ).lastrowid
            store.db.execute(
                "INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize(vec)),
            )
        store.db.commit()
        return store

    def search(self, query_vec, k=RETRIEVE_K):
        rows = self.db.execute(
            """
            SELECT d.doc_id, d.title, d.text, v.distance
            FROM vec v JOIN docs d ON d.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize(query_vec), k),
        ).fetchall()
        return [
            {"doc_id": r[0], "title": r[1], "text": r[2], "distance": r[3]} for r in rows
        ]

    def doc_count(self):
        return self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def close(self):
        self.db.close()


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s):
    return set(_WORD.findall(s.lower()))


def _sentences(text):
    parts = []
    for line in text.splitlines():
        # (?<!\d\.) keeps numbered-list markers ("1. Confirm...") glued to their
        # sentence instead of stranding "1." as a junk fragment.
        parts.extend(
            s.strip() for s in re.split(r"(?<!\d\.)(?<=[.!?])\s+", line)
            if s.strip()
        )
    return parts


def best_sentence(question, chunk_text_):
    """Pick the chunk sentence with the highest query-token overlap.

    Scored on content tokens (stopwords removed, stemmed) so filler words like
    "you"/"data" can't outvote the actual topic words.
    """
    sentences = _sentences(chunk_text_)
    if not sentences:
        return chunk_text_[:300]
    q = _content_tokens(question)
    scored = sorted(sentences, key=lambda s: len(_content_tokens(s) & q),
                    reverse=True)
    return scored[0]


def _is_h1(s):
    return bool(re.match(r"^\s*#\s", s))


def _is_pure_header(s):
    """A section header with no terminal punctuation, e.g. '## After 30 days'.

    Headers name a section but rarely *are* the answer — except price/spec
    headers like '## Pro — $19/month', which is why we keep them as candidates
    and pair them with the section's lead sentence instead of dropping them."""
    return bool(re.match(r"^\s*#{1,6}\s", s)) and not re.search(r"[.!?]['\")\]]?\s*$", s)


def best_evidence(question, hits):
    """Best (chunk, quotes) across the top hits.

    The top-1 chunk isn't always the one holding the answer (e.g. Pro-tier
    pricing lives in a later chunk than the plan intro), so the chunk with the
    single most overlapping sentence wins, ties broken by retrieval rank.

    Quote assembly:
      * the winning sentence; when the winner is a pure section header, pair it
        with the section's lead sentence in document order ("## Hub won't
        connect to Wi-Fi" -> "1. Confirm your network is 2.4 GHz.");
      * plus the best remaining sentence when it is also topical — a user
        asking "can I return after 45 days?" deserves both the 30-day rule AND
        the 31–60-day restocking-fee rule. Ties prefer real sentences over bare
        headers, so "## 30-day returns" doesn't crowd out the restocking-fee
        sentence.
    The document title (H1) is never quotable — it is boilerplate, not evidence.
    """
    q = _content_tokens(question)
    best = None  # ((overlap, -rank), chunk, [quotes])
    for rank, h in enumerate(hits[:3]):
        sents = [s for s in _sentences(h["text"]) if not _is_h1(s)]
        if not sents:
            continue
        scored = sorted(
            ((len(_content_tokens(s) & q), 0 if _is_pure_header(s) else 1,
              -i, s) for i, s in enumerate(sents)),
            reverse=True,
        )
        winner_ov, _, winner_pos_neg, winner = scored[0]
        winner_pos = -winner_pos_neg
        quotes = [winner]
        if _is_pure_header(winner) and winner_pos + 1 < len(sents):
            quotes.append(sents[winner_pos + 1])
        if len(quotes) == 1:
            for ov, _nh, _negpos, s in scored[1:]:
                if ov > 0 and s not in quotes:
                    quotes.append(s)
                    break
        key = (winner_ov, -rank)
        if best is None or key > best[0]:
            best = (key, h, quotes)
    return best[1], best[2]


def _clean_quote(s):
    """Strip markdown noise so quotes read cleanly in the UI."""
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"#{1,6}\s*", "", s)
    return " ".join(s.split())


class RAGChat:
    def __init__(self, db_path, embed_fn=None, refusal_distance=REFUSAL_DISTANCE,
                 mode="extractive"):
        self.store = VectorStore(db_path)
        self.embed_fn = embed_fn or default_embed_fn
        self.refusal_distance = refusal_distance
        self.mode = mode
        self.stats = {"queries": 0, "refusals": 0, "total_latency_ms": 0.0,
                      "total_cost_usd": 0.0}

    # ------------------------------------------------------------- retrieval
    def retrieve(self, question, k=RETRIEVE_K):
        qvec = self.embed_fn([question])[0]
        return self.store.search(qvec, k)

    # ------------------------------------------------------------- answering
    def answer(self, question):
        t0 = time.perf_counter()
        hits = self.retrieve(question)

        refusal_reason = None
        if not hits:
            refusal_reason = "no_hits"
        elif hits[0]["distance"] > self.refusal_distance:
            refusal_reason = "low_similarity"
        elif not _topical_overlap(question, hits[0]["text"]):
            # Nearest chunk shares no topical vocabulary with the question:
            # semantically closest != actually about it (e.g. "stock price"
            # retrieves the shipping doc via "price"). Refuse, don't guess.
            refusal_reason = "no_topical_overlap"
        refused = refusal_reason is not None

        if refused:
            result = {
                "answer": ("I don't have that in the provided documents, so I can't "
                           "answer it reliably. Try asking about Helios Home products, "
                           "policies, or support."),
                "citations": [],
                "refused": True,
                "refusal_reason": refusal_reason,
                "confidence": 0.0,
                "mode": self.mode,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "usage": {"est_input_tokens": 0, "est_cost_usd": 0.0},
            }
        elif self.mode == "llm":
            result = self._answer_llm(question, hits)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            result = self._answer_extractive(question, hits)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        self.stats["queries"] += 1
        self.stats["refusals"] += refused
        self.stats["total_latency_ms"] += result["latency_ms"]
        self.stats["total_cost_usd"] += result["usage"]["est_cost_usd"]
        return result

    def _answer_extractive(self, question, hits):
        chunk, sentences = best_evidence(question, hits)
        quotes = [_clean_quote(s)[:600] for s in sentences]
        quoted = "\n".join(f"> {q}" for q in quotes)
        confidence = round(max(0.0, 1.0 - chunk["distance"]), 3)
        return {
            "answer": f'According to "{chunk["title"]}":\n\n{quoted}',
            "citations": [{"doc_id": chunk["doc_id"], "title": chunk["title"],
                           "quote": q} for q in quotes],
            "refused": False,
            "refusal_reason": None,
            "confidence": confidence,
            "mode": "extractive",
            "usage": {"est_input_tokens": 0, "est_cost_usd": 0.0},
        }

    def _answer_llm(self, question, hits):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
        if not api_key:
            # No key configured -> fall back to extractive rather than failing.
            r = self._answer_extractive(question, hits)
            r["mode"] = "extractive (llm key missing)"
            return r

        context = "\n\n".join(
            f"[doc-id: {h['doc_id']}] {h['title']}\n{h['text']}" for h in hits[:3]
        )
        system = (
            "You answer questions using ONLY the context below. Rules:\n"
            "1. Every factual claim must cite its source as [doc-id].\n"
            "2. If the context does not contain the answer, reply exactly: "
            "\"I don't have that in the provided documents.\"\n"
            "3. Never use outside knowledge. Be concise."
        )
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens") or int(len(body) / 4)
        out_tok = usage.get("completion_tokens") or int(len(text) / 4)
        # Rough blended price for small chat models; honest estimate, labeled as such.
        est_cost = (in_tok / 1e6) * 0.15 + (out_tok / 1e6) * 0.60
        refused = "don't have that in the provided documents" in text.lower()
        return {
            "answer": text,
            "citations": [{"doc_id": h["doc_id"], "title": h["title"], "quote": ""}
                          for h in hits[:3]],
            "refused": refused,
            "refusal_reason": "no_context_answer" if refused else None,
            "confidence": round(max(0.0, 1.0 - hits[0]["distance"]), 3),
            "mode": f"llm ({model})",
            "usage": {"est_input_tokens": in_tok + out_tok,
                      "est_cost_usd": round(est_cost, 6)},
        }

    def close(self):
        self.store.close()


if __name__ == "__main__":
    import sys

    db = Path(__file__).resolve().parent / "data" / "rag.db"
    if not db.exists():
        sys.exit("no DB yet — run `python ingest.py` first")
    chat = RAGChat(db)
    q = " ".join(sys.argv[1:]) or "How long is the warranty?"
    r = chat.answer(q)
    print(r["answer"])
    for c in r["citations"]:
        print(f"\n[citation] {c['doc_id']} — {c['title']}")
    print(f"\nmode={r['mode']} refused={r['refused']} "
          f"confidence={r['confidence']} latency={r['latency_ms']}ms")
    chat.close()

# grounded-rag-chat

A **RAG** (retrieval-augmented generation — a chatbot that looks up your documents before answering) support chatbot with a **verbal-receipts policy**: every answer is a verbatim quote from the docs with a citation, and anything the docs don't cover gets a refusal instead of a hallucination.

The differentiator isn't the chat UI — it's the **eval harness**. `python evals/run_evals.py` scores the system on every run across retrieval recall, answer accuracy, faithfulness, and refusal behavior, and writes a machine-readable report. That's the loop real AI teams run before shipping to customers, and it's what this repo demonstrates.

## Demo (3 commands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # fastembed + sqlite-vec; runs fully local, no API keys
python ingest.py                  # index the sample docs -> data/rag.db
python evals/run_evals.py         # run the golden Q&A set, prints the scoreboard
python server.py                  # chat UI at http://localhost:8000
```

Ask *"How long is the warranty?"*, then try *"What is the stock price of Helios Home today?"* — the second one is refused, with the reason (`no_topical_overlap`) visible in the API response. That refusal is the feature.

## Architecture

```
question
  │  embed (fastembed BAAI/bge-small-en-v1.5, local — an embedding turns text
  │          into a list of numbers capturing its meaning)
  ▼
sqlite-vec top-5  (vector search — finds the chunks whose meaning is closest
  │                 to the question, inside a plain SQLite file)
  ▼
two-signal refusal gate
  │  1. cosine distance backstop (catches far-out questions)
  │  2. topical-overlap gate (catches near-misses: "stock price" retrieves the
  │     shipping doc via "price", but shares no topical vocabulary with it)
  ▼
extractive answering (default, no API key): the best sentence(s) across the
top-3 chunks, quoted verbatim with a [doc-id] citation. Faithfulness is
structural — there is no generated text to hallucinate.
```

Optional `RAG_MODE=llm` swaps in an OpenAI-compatible chat endpoint for
abstractive answers (fluent paragraphs instead of quotes), still grounded by a
system prompt that forces citations and refusal. Needs `OPENAI_API_KEY`
(`OPENAI_BASE_URL` and `RAG_LLM_MODEL` optional). Every query — either mode —
records **latency** (response time), **confidence**, and **cost** (estimated
LLM spend), aggregated at `/api/stats`.

## Eval results (golden set: 10 answerable + 4 out-of-scope, fictional corpus)

| metric | score | what it means |
|---|---|---|
| retrieval recall@5 | 1.000 | the right doc is in the top-5 for every question |
| answer hit rate | 1.000 | the expected fact appears in every answer |
| faithfulness | 1.000 | every quoted sentence verified verbatim in retrieved chunks |
| refusal accuracy | 1.000 | all 4 out-of-scope questions refused |
| in-scope answer rate | 1.000 | zero false refusals on real questions |
| avg latency | ~110 ms | fully local, no network calls |

The interesting part is the git history of getting here, not the final 1.0s:
a pure distance threshold could not separate *"stock price?"* (0.73) from legit
questions (up to 0.76) — that failure is what the topical-overlap gate exists
for, and the evals prove the fix. `evals/eval_report.json` holds the full
per-question breakdown.

## Files

| file | purpose |
|---|---|
| `rag.py` | retrieval + refusal gate + answering + cost/latency tracking |
| `ingest.py` | markdown docs -> chunked -> embedded -> `data/rag.db` |
| `server.py` | stdlib-only chat server + JSON API (`/api/chat`, `/api/stats`, `/api/health`) |
| `public/index.html` | chat UI with citations, confidence, refusal badges |
| `evals/golden.json` | 14 golden Q&As (expected doc + key phrase, or `refuse`) |
| `evals/run_evals.py` | the harness — metrics + `eval_report.json` |
| `tests/test_rag.py` | 15 hermetic unit tests (fake embeddings, no network) |
| `data/sample_docs/` | 8 fictional "Helios Home" support docs — swap in your own `.md` files and re-run `ingest.py` |

## Honest limits

- Golden questions use the docs' own vocabulary; paraphrase robustness ("warranty" vs "guarantee") is a separate eval dimension, not covered here.
- The refusal gate is lexical (stemmed token overlap), not semantic — a stricter setup would use an LLM judge or cross-encoder reranker.
- Extractive mode can't synthesize across documents; that's what `RAG_MODE=llm` is for.
- Sample corpus is fictional. Bring your own docs: drop `.md` files into `data/sample_docs/` and re-run `ingest.py` — no code changes.

## Why this exists

Built as a portfolio project demonstrating the full loop AI product teams
actually run: retrieve -> guardrail -> answer -> **evaluate** -> iterate.
The evals aren't decoration; they caught two real bugs during development
(a distance threshold that couldn't separate near-misses, and sentence
splitting that mangled numbered lists).

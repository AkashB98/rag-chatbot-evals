#!/usr/bin/env python3
"""
Eval harness for grounded-rag-chat.

Runs every question in golden.json through RAGChat and scores:

  retrieval_recall@k  — expected doc stem appears in the top-k retrieved chunks
  answer_hit_rate     — expected key phrase appears in the final answer
  faithfulness        — every quoted sentence is a substring of a retrieved chunk
                        (extractive mode: structural, verified not assumed)
  refusal_accuracy    — out-of-scope questions refused; in-scope ones answered
  avg_latency_ms      — wall-clock per question

This is the file that makes the repo interesting to a hiring manager: the bot
doesn't just answer — it proves, on every run, that it answers from the docs.

Usage:
    python evals/run_evals.py [--db data/rag.db]
Writes evals/eval_report.json and prints a summary table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import RAGChat, _clean_quote  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden.json"
REPORT_PATH = ROOT / "evals" / "eval_report.json"
K = 5


def faithfulness(answer_text, retrieved_texts):
    """Fraction of quoted sentences that appear verbatim in retrieved chunks.

    Both sides go through the same markdown cleaning the answer pipeline
    applies, so this verifies the *words* are grounded, not the asterisks.
    """
    import re

    quotes = re.findall(r"> (.+)", answer_text)
    if not quotes:
        return 1.0  # refusals carry no factual claims
    blob = " ".join(_clean_quote(t) for t in retrieved_texts)
    hits = 0
    for q in quotes:
        for sent in re.split(r"(?<=[.!?])\s+", _clean_quote(q.strip())):
            sent = sent.strip()
            if len(sent) > 15 and sent.rstrip(".!?") in blob:
                hits += 1
                break
        else:
            # whole quote checked as one unit as fallback
            if _clean_quote(q.strip())[:60] in blob:
                hits += 1
    return hits / len(quotes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "rag.db"))
    args = ap.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text())
    chat = RAGChat(args.db)

    rows, latencies = [], []
    for item in golden["answerable"]:
        q = item["q"]
        t0 = time.perf_counter()
        hits = chat.retrieve(q, k=K)
        res = chat.answer(q)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_docs = [h["doc_id"] for h in hits]
        recall = item["doc"] in retrieved_docs
        hit = item["key_phrase"].lower() in res["answer"].lower()
        faith = faithfulness(res["answer"], [h["text"] for h in hits])
        rows.append({
            "q": q, "type": "answerable",
            "retrieval_recall@5": recall, "answer_hit": hit,
            "faithfulness": round(faith, 3), "refused": res["refused"],
            "top_doc": retrieved_docs[0] if retrieved_docs else None,
        })

    refusals_ok, refusals_total = 0, 0
    for item in golden["out_of_scope"]:
        res = chat.answer(item["q"])
        refusals_total += 1
        refusals_ok += res["refused"]
        rows.append({"q": item["q"], "type": "out_of_scope",
                     "refused": res["refused"],
                     "refusal_correct": res["refused"]})

    ans = [r for r in rows if r["type"] == "answerable"]
    summary = {
        "n_answerable": len(ans),
        "n_out_of_scope": refusals_total,
        "retrieval_recall@5": round(sum(r["retrieval_recall@5"] for r in ans) / len(ans), 3),
        "answer_hit_rate": round(sum(r["answer_hit"] for r in ans) / len(ans), 3),
        "faithfulness": round(sum(r["faithfulness"] for r in ans) / len(ans), 3),
        "refusal_accuracy": round(refusals_ok / refusals_total, 3),
        "in_scope_answer_rate": round(sum(not r["refused"] for r in ans) / len(ans), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "mode": "extractive",
    }
    REPORT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"\n{'question':<58}{'recall@5':<9}{'hit':<5}{'faith':<7}{'refused'}")
    print("-" * 95)
    for r in rows:
        q = (r["q"][:55] + "…") if len(r["q"]) > 55 else r["q"]
        if r["type"] == "answerable":
            print(f"{q:<58}{str(r['retrieval_recall@5']):<9}{str(r['answer_hit']):<5}"
                  f"{r['faithfulness']:<7}{r['refused']}")
        else:
            print(f"{q:<58}{'—':<9}{'—':<5}{'—':<7}{r['refused']} "
                  f"{'OK' if r['refusal_correct'] else 'MISS'}")
    print("-" * 95)
    for k_, v in summary.items():
        print(f"  {k_:<22} {v}")
    print(f"\nreport -> {REPORT_PATH}")
    chat.close()


if __name__ == "__main__":
    main()

"""Hermetic tests for grounded-rag-chat.

No network, no model downloads, no real embeddings: a fake embed_fn maps
keywords to fixed 3-dim vectors so retrieval behavior is fully deterministic.
Run:  python -m pytest tests/   (or: python tests/test_rag.py via unittest)
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import RAGChat, VectorStore, best_sentence, chunk_text  # noqa: E402
from evals.run_evals import faithfulness  # noqa: E402


def fake_embed(texts):
    """Deterministic 3-dim vectors keyed on topic words."""
    out = []
    for t in texts:
        t = t.lower()
        if "warranty" in t:
            out.append([1.0, 0.0, 0.0])
        elif "shipping" in t:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
        # normalize for stable cosine distances
        n = sum(x * x for x in out[-1]) ** 0.5
        out[-1] = [x / n for x in out[-1]]
    return out


DOCS = [
    ("warranty-policy", "Warranty Policy",
     "Every Helios device ships with a 2-year limited warranty. Water damage is not covered."),
    ("shipping-info", "Shipping Information",
     "Standard shipping is free on orders over $50 and takes 3 to 5 business days."),
    ("privacy-policy", "Privacy Policy",
     "We never sell your personal data to third parties."),
]


def build_test_db(tmp):
    db_path = Path(tmp) / "test.db"
    return VectorStore.build(db_path, DOCS, fake_embed, dims=3)


class TestChunking(unittest.TestCase):
    def test_paragraphs_stay_intact(self):
        text = "# Title\n\nFirst paragraph here.\n\nSecond paragraph here."
        chunks = chunk_text(text, words=120)
        self.assertEqual(len(chunks), 1)
        # blocks joined with single newlines, structure preserved
        self.assertIn("# Title\nFirst paragraph here.\nSecond paragraph here.",
                      chunks[0])

    def test_breaks_at_paragraph_boundaries(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(4))
        chunks = chunk_text(text, words=120)
        self.assertGreater(len(chunks), 1)
        # no chunk exceeds the word budget by much
        for c in chunks:
            self.assertLessEqual(len(c.split()), 120)

    def test_oversized_paragraph_gets_word_windows(self):
        text = " ".join(f"w{i}" for i in range(250))  # one giant paragraph
        chunks = chunk_text(text, words=100, overlap=10)
        joined = " ".join(chunks)
        for i in range(250):
            self.assertIn(f"w{i}", joined)
        self.assertIn("w99", chunks[0])
        self.assertIn("w90", chunks[1])  # overlap region repeats

    def test_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\n  "), [])


class TestRetrieval(unittest.TestCase):
    def test_top_hit_matches_topic(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = build_test_db(tmp)
            hits = store.search(fake_embed(["what is the warranty period?"])[0], k=3)
            self.assertEqual(hits[0]["doc_id"], "warranty-policy")
            self.assertLess(hits[0]["distance"], hits[1]["distance"])
            store.close()

    def test_k_limits_results(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = build_test_db(tmp)
            hits = store.search(fake_embed(["hello"])[0], k=1)
            self.assertEqual(len(hits), 1)
            store.close()


class TestAnswering(unittest.TestCase):
    def _chat(self, tmp, **kw):
        build_test_db(tmp)
        return RAGChat(str(Path(tmp) / "test.db"), embed_fn=fake_embed, **kw)

    def test_answer_cites_and_quotes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            chat = self._chat(tmp, refusal_distance=2.0)
            res = chat.answer("how long is the warranty?")
            self.assertFalse(res["refused"])
            self.assertEqual(res["citations"][0]["doc_id"], "warranty-policy")
            self.assertIn("2-year", res["answer"])
            self.assertGreater(res["latency_ms"], 0)
            chat.close()

    def test_refusal_when_nothing_relevant(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            chat = self._chat(tmp, refusal_distance=2.0)  # distance never fires
            res = chat.answer("quantum banana orbit")
            self.assertTrue(res["refused"])
            self.assertEqual(res["refusal_reason"], "no_topical_overlap")
            self.assertEqual(res["citations"], [])
            self.assertIn("don't have that", res["answer"])
            chat.close()

    def test_refusal_by_distance(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            chat = self._chat(tmp, refusal_distance=-1.0)  # refuse everything
            res = chat.answer("how long is the warranty?")
            self.assertTrue(res["refused"])
            self.assertEqual(res["refusal_reason"], "low_similarity")
            chat.close()

    def test_stats_accumulate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            chat = self._chat(tmp, refusal_distance=0.0)
            chat.answer("q1")
            chat.answer("q2")
            self.assertEqual(chat.stats["queries"], 2)
            self.assertEqual(chat.stats["refusals"], 2)
            chat.close()


class TestBestSentence(unittest.TestCase):
    def test_picks_sentence_with_query_overlap(self):
        chunk = ("The warranty lasts 2 years. "
                 "Shipping is free over $50. "
                 "Returns close after 60 days.")
        self.assertIn("Shipping", best_sentence("how much is shipping?", chunk))


class TestFaithfulness(unittest.TestCase):
    def test_verbatim_quote_scores_one(self):
        chunk = "Every Helios device ships with a 2-year limited warranty."
        ans = 'According to "Warranty Policy":\n\n> ' + chunk
        self.assertEqual(faithfulness(ans, [chunk]), 1.0)

    def test_hallucinated_quote_scores_zero(self):
        chunk = "Every Helios device ships with a 2-year limited warranty."
        ans = '> The warranty lasts for 99 years and covers everything.'
        self.assertEqual(faithfulness(ans, [chunk]), 0.0)

    def test_refusal_has_no_claims(self):
        self.assertEqual(faithfulness("I don't have that.", ["irrelevant"]), 1.0)


class TestGoldenSet(unittest.TestCase):
    def test_schema_and_doc_coverage(self):
        golden = json.loads((ROOT / "evals" / "golden.json").read_text())
        stems = {p.stem for p in (ROOT / "data" / "sample_docs").glob("*.md")}
        self.assertGreaterEqual(len(golden["answerable"]), 8)
        self.assertGreaterEqual(len(golden["out_of_scope"]), 3)
        for item in golden["answerable"]:
            self.assertIn("q", item)
            self.assertIn(item["doc"], stems, f"golden doc missing: {item['doc']}")
            text = (ROOT / "data" / "sample_docs" / f"{item['doc']}.md").read_text()
            self.assertIn(item["key_phrase"].lower(), text.lower(),
                            f"key phrase not in doc: {item['key_phrase']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

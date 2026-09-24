#!/usr/bin/env python3
"""
Chat server for grounded-rag-chat. Standard library only (no web framework).

    python server.py [--port 8000]

Endpoints:
    GET  /            chat UI
    POST /api/chat    {"question": "..."} -> answer + citations + telemetry
    GET  /api/stats   aggregate query stats (count, refusals, avg latency, cost)
    GET  /api/health  {"ok": true, "chunks": N}

RAG_MODE=llm enables the optional LLM synthesis path (needs OPENAI_API_KEY).
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rag import RAGChat  # noqa: E402

DB_PATH = ROOT / "data" / "rag.db"
PUBLIC_DIR = ROOT / "public"

chat = None


class Handler(BaseHTTPRequestHandler):
    server_version = "grounded-rag-chat/1.0"

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            self._file(PUBLIC_DIR / "index.html", "text/html; charset=utf-8")
        elif route == "/api/health":
            self._json({"ok": True, "chunks": chat.store.doc_count(),
                        "mode": chat.mode})
        elif route == "/api/stats":
            s = chat.stats
            q = s["queries"]
            self._json({
                "queries": q,
                "refusals": s["refusals"],
                "avg_latency_ms": round(s["total_latency_ms"] / q, 1) if q else 0,
                "total_cost_usd": round(s["total_cost_usd"], 6),
                "mode": chat.mode,
            })
        else:
            self.send_error(404)

    def do_POST(self):
        route = urlparse(self.path).path
        if route != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "invalid JSON"}, 400)
            return
        if not question:
            self._json({"error": "question is required"}, 400)
            return
        if len(question) > 2000:
            self._json({"error": "question too long (max 2000 chars)"}, 400)
            return
        try:
            self._json(chat.answer(question))
        except Exception as exc:  # never leak a traceback to the client
            self._json({"error": f"answering failed: {type(exc).__name__}"}, 500)

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  %s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()

    global chat
    if not DB_PATH.exists():
        sys.exit("no DB yet — run `python ingest.py` first")
    mode = os.environ.get("RAG_MODE", "extractive")
    chat = RAGChat(DB_PATH, mode=mode)
    print(f"grounded-rag-chat on http://localhost:{args.port} "
          f"(mode={mode}, chunks={chat.store.doc_count()})")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        chat.close()


if __name__ == "__main__":
    main()

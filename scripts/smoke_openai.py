"""SYNTH_O1 smoke: the OpenAI-compatible backend proven against
a REAL local HTTP server (stdlib, canned responses) — request
shape, auth header, system placement, both response dialects,
error wrapping — then the full render pipeline through it.

Run from the repo root:

    python scripts/smoke_openai.py
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.openai_compat import (OpenAICompatBackend,
                                    OpenAICompatError)

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


RECEIVED = []
REPLIES = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        RECEIVED.append({"path": self.path, "body": body,
                         "auth": self.headers.get(
                             "Authorization", "")})
        status, payload = REPLIES.pop(0) if REPLIES else \
            (200, {"choices": [{"message":
                                {"content": "pong"}}]})
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever,
                     daemon=True).start()
    base = "http://127.0.0.1:{}/v1".format(server.server_port)

    b = OpenAICompatBackend(base_url=base, model="test-model",
                            api_key="sk-demo")
    REPLIES.append((200, {"choices": [{"message": {
        "content": "hello from the local server"}}]}))
    text = b.complete("the prompt", system="the system",
                      max_tokens=77, temperature=0.25)
    call = RECEIVED[-1]
    check("requests hit /chat/completions with model, "
          "max_tokens, and temperature",
          call["path"].endswith("/chat/completions")
          and call["body"]["model"] == "test-model"
          and call["body"]["max_tokens"] == 77
          and call["body"]["temperature"] == 0.25)
    check("system prompts travel as the leading system message",
          call["body"]["messages"][0] == {
              "role": "system", "content": "the system"}
          and call["body"]["messages"][1]["content"]
          == "the prompt")
    check("api keys travel as a Bearer header",
          call["auth"] == "Bearer sk-demo")
    check("chat-dialect responses are extracted",
          text == "hello from the local server")

    REPLIES.append((200, {"choices": [
        {"text": "legacy completion dialect"}]}))
    check("legacy text-dialect responses are extracted too",
          OpenAICompatBackend(base_url=base).complete("p")
          == "legacy completion dialect")

    b2 = OpenAICompatBackend(base_url=base)
    REPLIES.append((200, {"choices": [{"message":
                                       {"content": "x"}}]}))
    b2.complete("p")
    check("no key means no Authorization header",
          RECEIVED[-1]["auth"] == "")

    REPLIES.append((200, {"unexpected": True}))
    try:
        OpenAICompatBackend(base_url=base).complete("p")
        check("malformed responses wrap into "
              "OpenAICompatError", False)
    except OpenAICompatError:
        check("malformed responses wrap into "
              "OpenAICompatError", True)

    try:
        OpenAICompatBackend(
            base_url="http://127.0.0.1:9/v1",
            timeout_s=2).complete("p")
        check("unreachable servers raise a llamafile-citing "
              "hint", False)
    except OpenAICompatError as e:
        check("unreachable servers raise a llamafile-citing "
              "hint", "llamafile" in str(e))

    # The pipeline in anger: render a small corpus through the
    # fake server; hopeless replies exercise verify-fallback.
    from synthkit.corpus_io import write_corpus
    from synthkit.examples import reference_spec
    from synthkit.planner import plan_corpus
    from synthkit.renderer import render_corpus
    spec = reference_spec(size=3)
    blueprints = plan_corpus(spec)
    for _ in range(40):
        REPLIES.append((200, {"choices": [{"message": {
            "content": "a note with none of the needles"}}]}))
    backend = OpenAICompatBackend(base_url=base)
    documents, report = render_corpus(spec, blueprints, backend)
    check("the render pipeline runs unchanged over the "
          "openai-compat backend, fallback keeping it safe",
          len(documents) == 3
          and report.fallbacks == 3)

    from synthkit.cli import _backend
    bb = _backend("openai", "somemodel")
    check("the CLI backend registry knows `openai`",
          bb.name == "openai" and bb.model == "somemodel")

    server.shutdown()
    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()

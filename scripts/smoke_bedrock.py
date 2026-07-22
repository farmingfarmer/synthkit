"""SYNTH_K1 smoke: the Bedrock backend proven without AWS — a
fake injected client exercises the full request/response path:
payload shape, system prompt placement, temperature passthrough,
response parsing, and error wrapping. The same backend then
satisfies the compiler contract end to end via a scripted
compile.

Run from the repo root:

    python scripts/smoke_bedrock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.bedrock import BedrockBackend, BedrockError
from synthkit.compiler import compile_table_spec
from synthkit.examples import reference_table

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        text = self.payloads.pop(0)
        if isinstance(text, Exception):
            raise text
        return {"output": {"message": {"content": [
            {"text": text}]}}}


def main():
    try:
        BedrockBackend("")
        check("empty model_id refused with an example", False)
    except BedrockError as e:
        check("empty model_id refused with an example",
              "e.g." in str(e))

    fake = FakeClient(["hello from the cloud"])
    backend = BedrockBackend("anthropic.claude-test",
                             client=fake)
    text = backend.complete("the prompt", system="the system",
                            max_tokens=123, temperature=0.5)
    call = fake.calls[0]
    check("converse payload carries model, message, and "
          "inference config faithfully",
          call["modelId"] == "anthropic.claude-test"
          and call["messages"][0]["content"][0]["text"]
          == "the prompt"
          and call["inferenceConfig"]["maxTokens"] == 123
          and call["inferenceConfig"]["temperature"] == 0.5)
    check("system prompts travel in the system block",
          call["system"] == [{"text": "the system"}])
    check("response text is extracted across content parts",
          text == "hello from the cloud")

    fake = FakeClient(["no system this time"])
    BedrockBackend("m", client=fake).complete("p")
    check("empty system prompts are omitted, not sent blank",
          "system" not in fake.calls[0])

    fake = FakeClient([RuntimeError("throttled")])
    try:
        BedrockBackend("m", client=fake).complete("p")
        check("client failures wrap into BedrockError", False)
    except BedrockError as e:
        check("client failures wrap into BedrockError",
              "throttled" in str(e))

    class WeirdClient:
        def converse(self, **kwargs):
            return {"unexpected": True}

    try:
        BedrockBackend("m", client=WeirdClient()).complete("p")
        check("malformed responses wrap into BedrockError",
              False)
    except BedrockError:
        check("malformed responses wrap into BedrockError", True)

    # The contract in anger: a scripted compile through Bedrock.
    good = reference_table(rows=40).to_json()
    fake = FakeClient([good])
    result = compile_table_spec("forty rows please",
                                BedrockBackend("m", client=fake))
    check("the compiler runs unchanged over a Bedrock backend",
          result.ok
          and "table architect" not in good)  # sanity: spec text

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()

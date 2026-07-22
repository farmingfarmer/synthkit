"""SYNTH_K1: the Bedrock backend — synthkit on work
infrastructure.

Implements the same backend contract as OllamaBackend
(`complete(prompt, system=..., max_tokens=..., temperature=...)`)
over the AWS Bedrock Converse API, so the compiler, the renderer,
and the LLM-as-vendor adapter all take it unchanged:

    from synthkit.bedrock import BedrockBackend
    backend = BedrockBackend(model_id="anthropic.claude-3-...")
    compile_table_spec(description, backend)

Design for testability: the boto3 client is INJECTED (built
lazily from boto3 only when not provided), so the full
request/response path is smoke-tested against a fake client with
no AWS account in sight — the same discipline as the canned
compiler backends. boto3 itself is imported only on first real
use; synthkit's core stays stdlib-pure.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Optional


class BedrockError(RuntimeError):
    pass


class BedrockBackend:
    name = "bedrock"

    def __init__(self, model_id: str,
                 client: Optional[object] = None,
                 region: str = "us-west-2"):
        if not model_id:
            raise BedrockError("model_id is required — e.g. "
                               "'anthropic.claude-3-5-sonnet-"
                               "20240620-v1:0'")
        self.model_id = model_id
        self.region = region
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise BedrockError(
                    "boto3 is not installed — `pip install "
                    "boto3` (SageMaker kernels ship it)")
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region)
        return self._client

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 3000,
                 temperature: float = 0.0) -> str:
        client = self._get_client()
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user",
                          "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        try:
            response = client.converse(**kwargs)
        except Exception as e:
            raise BedrockError(
                "Bedrock converse failed for {}: {}".format(
                    self.model_id, e))
        try:
            parts = response["output"]["message"]["content"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, TypeError) as e:
            raise BedrockError(
                "unexpected Bedrock response shape: {}".format(e))

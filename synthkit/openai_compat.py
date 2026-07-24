"""SYNTH_O1: the OpenAI-compatible backend — synthkit against
any open-source model server.

llamafile, LM Studio, vLLM, TGI, LiteLLM, Ollama's /v1 route,
and most enterprise LLM gateways all serve the same dialect:
POST {base_url}/chat/completions. One backend covers the entire
open-source serving landscape:

    from synthkit.openai_compat import OpenAICompatBackend
    backend = OpenAICompatBackend(
        base_url="http://127.0.0.1:8080/v1",
        model="local")            # llamafile ignores the name

CLI:  --backend openai --model <name>
      base url from SYNTHKIT_OPENAI_BASE
      (default http://127.0.0.1:8080/v1); api key, if the
      server wants one, from SYNTHKIT_OPENAI_KEY.

Design mirrors the Bedrock backend: the HTTP transport is
injectable, so the full request/response path is smoke-tested
against a local fake server with no model in sight. stdlib only;
Python 3.8 compatible.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class OpenAICompatError(RuntimeError):
    pass


DEFAULT_BASE = "http://127.0.0.1:8080/v1"


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, base_url: str = "",
                 model: str = "local",
                 api_key: str = "",
                 timeout_s: float = 300.0,
                 opener=None):
        self.base_url = (base_url
                         or os.environ.get(
                             "SYNTHKIT_OPENAI_BASE", "")
                         or DEFAULT_BASE).rstrip("/")
        self.model = model or "local"
        self.api_key = api_key or os.environ.get(
            "SYNTHKIT_OPENAI_KEY", "")
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 3000,
                 temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system",
                             "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body, headers=headers)
        try:
            with self._opener(req,
                              timeout=self.timeout_s) as resp:
                payload = json.loads(
                    resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OpenAICompatError(
                "cannot reach {} — is the model server "
                "running? (llamafile: `llamafile.exe -m "
                "model.gguf --server --port 8080`) [{}]".format(
                    self.base_url, e))
        except Exception as e:
            raise OpenAICompatError(
                "openai-compat call failed: {}".format(e))
        try:
            choice = payload["choices"][0]
            msg = choice.get("message") or {}
            text = msg.get("content")
            if text is None:
                text = choice.get("text")
            if text is None:
                raise KeyError("no content")
            return text
        except (KeyError, IndexError, TypeError) as e:
            raise OpenAICompatError(
                "unexpected response shape from {}: {}".format(
                    self.base_url, e))

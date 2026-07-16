"""SYNTH_V1 S1: backends and the SpecCompiler.

The LLMBackend protocol is the deployment seam: development runs
Ollama on the M3; Keck runs Bedrock or the Anthropic API. Every
LLM-touching stage takes a backend instance, so the move is a config
change, not a port.

The compiler is LLM-ASSISTED, not LLM-trusted: plain English goes
in, a draft DataSpec comes out, validation runs, and the human
reviews the JSON before anything generates. A compile that fails
validation returns the problems alongside the draft — the fix loop
is human-in-the-middle by design.

Python 3.8 compatible. Stdlib only (backends import their SDKs
lazily so the library works wherever at least one is available).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .spec import DataSpec, SpecError

log = logging.getLogger(__name__)


# ===================================================================
# Backends
# ===================================================================

class LLMBackend:
    """Protocol: complete(prompt, system, max_tokens, temperature)
    -> str. Adapters raise BackendError on hard failure."""

    name = "base"

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        raise NotImplementedError


class BackendError(RuntimeError):
    pass


class OllamaBackend(LLMBackend):
    """Local models via the Ollama HTTP API (development default)."""

    name = "ollama"

    def __init__(self, model: str = "mistral-small3.1",
                 host: str = "http://localhost:11434",
                 timeout_s: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature,
                        "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req,
                                        timeout=self.timeout_s) as r:
                return json.loads(r.read().decode("utf-8")).get(
                    "response", "")
        except Exception as e:
            raise BackendError("ollama call failed: {}".format(e))


class AnthropicBackend(LLMBackend):
    """Anthropic API (lazy SDK import)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        try:
            import anthropic
        except ImportError:
            raise BackendError("anthropic SDK not installed")
        client = anthropic.Anthropic(api_key=self.api_key) \
            if self.api_key else anthropic.Anthropic()
        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or None,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                b.text for b in msg.content
                if getattr(b, "type", "") == "text")
        except Exception as e:
            raise BackendError("anthropic call failed: {}".format(e))


class HFLocalBackend(LLMBackend):
    """IN-PROCESS open-source model via Hugging Face transformers —
    generation happens inside the notebook's own Python process, no
    external server or service.

        pip install transformers torch accelerate

    Weights download from the Hugging Face hub on first use (or pass
    a local path / S3-synced directory as model_id for air-gapped
    environments). Small instruct models are the sweet spot for
    synthkit's short verified renders:

        HFLocalBackend("Qwen/Qwen2.5-1.5B-Instruct")   # ~3GB
        HFLocalBackend("Qwen/Qwen2.5-0.5B-Instruct")   # ~1GB, CPU-ok

    The renderer's verifier + retry + fallback wrap this like any
    backend: a weak model degrades measurably, never breaks the
    corpus. `pipeline` is injectable for tests.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device_map: str = "auto",
                 pipeline: Any = None):
        self.model_id = model_id
        self.name = "hf-local/" + model_id
        self._device_map = device_map
        self._pipe = pipeline

    def _ensure_pipe(self) -> Any:
        if self._pipe is None:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError:
                raise BackendError(
                    "transformers is not installed — run: pip "
                    "install transformers torch accelerate")
            try:
                self._pipe = hf_pipeline(
                    "text-generation", model=self.model_id,
                    device_map=self._device_map)
            except Exception as e:
                raise BackendError(
                    "could not load {}: {}".format(self.model_id, e))
        return self._pipe

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        pipe = self._ensure_pipe()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {"max_new_tokens": max_tokens,
                  "do_sample": temperature > 0,
                  "temperature": max(temperature, 0.01),
                  "return_full_text": False}
        try:
            out = pipe(messages, **kwargs)
        except (TypeError, ValueError):
            text_in = (system + "\n\n" + prompt) if system else prompt
            try:
                out = pipe(text_in, **kwargs)
            except Exception as e:
                raise BackendError("hf-local generation failed: "
                                   "{}".format(e))
        except Exception as e:
            raise BackendError("hf-local generation failed: "
                               "{}".format(e))
        try:
            text = out[0]["generated_text"]
            if isinstance(text, list):
                text = text[-1].get("content", "")
            return str(text)
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise BackendError(
                "hf-local returned an unexpected shape: {}".format(e))


class BedrockBackend(LLMBackend):
    """AWS Bedrock via boto3 (the Keck deployment path).
    Anthropic-on-Bedrock message format."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str = "us-west-2"):
        self.model_id = model_id
        self.region = region

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        try:
            import boto3
        except ImportError:
            raise BackendError("boto3 not installed")
        client = boto3.client("bedrock-runtime",
                              region_name=self.region)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        try:
            resp = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body))
            parsed = json.loads(resp["body"].read())
            return "".join(
                b.get("text", "") for b in parsed.get("content", []))
        except Exception as e:
            raise BackendError("bedrock call failed: {}".format(e))


# ===================================================================
# The compiler
# ===================================================================

_COMPILER_SYSTEM = """You compile a plain-English description of a \
synthetic dataset into a strict JSON DataSpec. Output ONLY the JSON \
object, no prose, no markdown fences, starting with '{'.

Schema (all keys required unless noted):
{
  "title": "short name for the dataset",
  "structured_fields": [
    {"name": "field_name",
     "ftype": "int|float|str|date|categorical|id|person_name",
     "distribution": {"kind": "uniform|normal|lognormal|categorical|date_range|sequence",
                      "params": { ... kind-specific ... }},
     "nullable_rate": 0.0}
  ],
  "unstructured_fields": [
    {"name": "field_name", "note_type": "what kind of note this is",
     "target_elements": [
       {"element_id": "snake_case_id",
        "description": "the fact an extractor should find",
        "phrasings": ["two or more surface forms, use {value} where a sampled value goes"],
        "density": 0.8, "difficulty": "easy|medium|hard",
        "value_source": "structured_field_name OR {\\"choices\\": [..]} OR null"}
     ],
     "distractors": [
       {"distractor_id": "snake_case_id",
        "description": "plausible near-miss that should NOT be extracted",
        "phrasings": ["..."], "density": 0.5, "value_source": null}
     ],
     "style": {"personas": ["..."], "verbosity": ["terse","moderate","verbose"],
               "abbreviation": ["none","moderate","heavy"]},
     "length_words": [80, 220]}
  ],
  "cross_field_rules": [
    {"earlier": "field_a", "later": "field_b",
     "min_delta": 0, "max_delta": 30}
  ],
  "corpus": {"size": 50, "master_seed": 20260715}
}

Distribution params: uniform {"low","high"}; normal {"mean","stdev",
optional "min","max"}; lognormal {"mu","sigma",optional "min","max"};
categorical {"choices",optional "weights"}; date_range {"start","end"
as YYYY-MM-DD}; sequence {"prefix","start"} for ids.

Rules:
- Every target element gets >=2 phrasings with genuinely different
  surface structure.
- Include distractors: plausible near-misses of the targets.
- densities in (0,1]; difficulties spread across easy/medium/hard.
- NEVER invent real-sounding institutions, patients, or providers;
  values are generic and synthetic.
- Sizes and seeds: honor the user's numbers; default size 50."""


@dataclass
class CompileResult:
    spec: Optional[DataSpec]
    raw_json: str
    problems: Optional[str]      # None when the spec validates

    @property
    def ok(self) -> bool:
        return self.spec is not None and self.problems is None


def _extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def compile_spec(description: str, backend: LLMBackend,
                 retries: int = 1) -> CompileResult:
    """Plain English -> validated DataSpec (or draft + problems).
    On validation failure, one repair round feeds the problems back
    to the model; after that, the human takes over — by design."""
    prompt = ("Dataset description:\n\n{}\n\n=== END DESCRIPTION ===\n"
              "Now output ONLY the DataSpec JSON object, starting "
              "with '{{'.".format(description.strip()))
    last_raw = ""
    last_problems = "no output produced"
    for attempt in range(retries + 1):
        raw = backend.complete(prompt, system=_COMPILER_SYSTEM,
                               max_tokens=3000, temperature=0.3)
        json_str = _extract_json(raw)
        if json_str is None:
            last_raw, last_problems = raw or "", "no JSON object in output"
        else:
            last_raw = json_str
            try:
                spec = DataSpec.from_json(json_str)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                last_problems = "JSON did not fit the schema: {}".format(e)
            else:
                try:
                    spec.validate()
                    return CompileResult(spec=spec, raw_json=json_str,
                                         problems=None)
                except SpecError as e:
                    last_problems = str(e)
                    log.info("compile attempt %d: %d validation "
                             "problem(s)", attempt + 1,
                             last_problems.count("\n") + 1)
        if attempt < retries:
            prompt = (
                "Dataset description:\n\n{}\n\n"
                "Your previous DataSpec had these problems:\n{}\n\n"
                "Output the CORRECTED DataSpec JSON only, starting "
                "with '{{'.".format(description.strip(), last_problems)
            )
    try:
        draft = DataSpec.from_json(last_raw)
    except Exception:
        draft = None
    return CompileResult(spec=draft, raw_json=last_raw,
                         problems=last_problems)

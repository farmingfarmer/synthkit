"""Put a language model in the chair and grade it on clinical
notes — by corruption type, not by a single number.

The contract asked of the model is deliberately narrow: for each
named fact, say whether the note asserts it is PRESENT, whether
that assertion is CURRENT or historical, and whether it is CERTAIN
or hedged. Those three questions are exactly where clinical
extraction goes wrong, and grading them separately turns "recall
0.83" into "recall 0.83, but it takes every hedged finding as
confirmed and reverses every compound negation" — which names a
competence rather than issuing a mark.

The prompt is BLIND: the model is told which facts to look for and
nothing about how the text was corrupted. Prompt hardening goes
through `extra_system`, so an intervention is explicit, named, and
diffable against the base run — the same seam the earlier vendor
study used to take a trap rate from 26% to under 4%.

Unusable replies are counted, never crashed on. A model that
cannot follow the answer format is not a broken harness; it is a
finding, and one worth reporting beside its accuracy.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llmvendor import _find_json_array

SYSTEM = """You are extracting structured facts from a clinical \
note.

For each fact in the list, decide what the NOTE ITSELF asserts —
not what is clinically likely, and not what you would infer.

Return ONLY a JSON array. One object per fact you can find:
  {"fact": "<exact fact key>",
   "present": true|false,
   "current": true|false,
   "certain": true|false,
   "value": "<number if the fact is a measurement, else null>"}

  present: does the note assert this finding is TRUE of the
           patient? A negated mention means present=false.
  current: does it describe the patient NOW? A historical or
           carried-forward mention means current=false.
  certain: is it asserted plainly? A hedged mention
           ("possible", "cannot rule out") means certain=false.

Omit any fact the note does not mention at all. Return the array \
and nothing else — no prose, no code fences."""

TASK = """Facts to look for:
{facts}

Clinical note:
---
{note}
---

JSON array:"""


class NoteExtractor:
    """A language model adapted to the note-grading contract."""

    def __init__(self, backend, facts: List[Dict[str, str]],
                 name: str = "llm-note-vendor",
                 temperature: float = 0.0,
                 max_tokens: int = 800,
                 extra_system: str = ""):
        self.backend = backend
        self.facts = facts
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_system = (extra_system or "").strip()
        self.system = SYSTEM
        if self.extra_system:
            self.system += "\n\n" + self.extra_system
        self.valid_keys = {f["key"] for f in facts}
        self.stats: Dict[str, int] = {
            "calls": 0, "malformed": 0, "empty": 0,
            "unknown_fact": 0, "bad_item": 0}

    def _fact_list(self) -> str:
        return "\n".join(
            "- {} : {}".format(f["key"], f["label"])
            for f in self.facts)

    def extract(self, note: str) -> List[Dict[str, Any]]:
        self.stats["calls"] += 1
        raw = self.backend.complete(
            TASK.format(facts=self._fact_list(), note=note),
            system=self.system, max_tokens=self.max_tokens,
            temperature=self.temperature)
        arr = _find_json_array(raw or "")
        if arr is None:
            # nothing salvageable — counted, not crashed on
            self.stats["malformed"] += 1
            return []
        if not arr:
            self.stats["empty"] += 1
            return []
        out: List[Dict[str, Any]] = []
        for item in arr:
            if not isinstance(item, dict):
                self.stats["bad_item"] += 1
                continue
            key = str(item.get("fact", "")).strip()
            if key not in self.valid_keys:
                # a hallucinated fact key is its own failure mode
                self.stats["unknown_fact"] += 1
                continue
            out.append({
                "fact": key,
                "present": _as_bool(item.get("present"), True),
                "current": _as_bool(item.get("current"), True),
                "certain": _as_bool(item.get("certain"), True),
                "value": (None if item.get("value") in
                          (None, "", "null")
                          else str(item.get("value")).strip()),
            })
        return out

    def extract_all(self, notes: List[str]
                    ) -> List[List[Dict[str, Any]]]:
        return [self.extract(n) for n in notes]

    def reliability(self) -> Dict[str, Any]:
        c = max(self.stats["calls"], 1)
        bad = self.stats["malformed"]
        return {
            "name": self.name,
            "calls": self.stats["calls"],
            "malformed": bad,
            "malformed_rate": round(bad / c, 4),
            "empty_replies": self.stats["empty"],
            "hallucinated_fact_keys": self.stats["unknown_fact"],
            "non_object_items": self.stats["bad_item"],
            "reading": "a model that cannot hold the answer "
                       "format is not a harness problem — it is a "
                       "procurement finding, and it belongs beside "
                       "the accuracy numbers rather than hidden "
                       "behind them",
        }


def _as_bool(v, default=True) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "present"):
        return True
    if s in ("false", "no", "n", "0", "absent"):
        return False
    return default


def facts_from_specs(fact_specs) -> List[Dict[str, str]]:
    """Turn the transcriber's FactSpec list into the prompt's
    fact list, so the model is asked about exactly what was
    planted — no more, no less."""
    return [{"key": f.column, "label": f.label}
            for f in fact_specs
            if f.placement != "structured_only"]


class ScriptedBackend:
    """A deterministic stand-in for tests and dry runs.

    `behaviour` selects a failure mode to imitate, so the grading
    path can be exercised — and proven to detect each failure —
    without a model or a network.
    """

    def __init__(self, facts, behaviour: str = "careful",
                 terms: Optional[Dict[str, List[str]]] = None):
        self.facts = facts
        self.behaviour = behaviour
        self.terms = terms or {}
        self.calls = 0

    def complete(self, prompt, system="", max_tokens=0,
                 temperature=0.0):
        import re
        self.calls += 1
        note = prompt.split("---")[1] if "---" in prompt else prompt
        if self.behaviour == "malformed":
            return "Sure! Here are the findings I noticed:"
        out = []
        low = note.lower()
        for f in self.facts:
            terms = self.terms.get(f["key"], [f["label"]])
            hit = None
            for t in terms:
                if t.lower() in low:
                    hit = t
                    break
            if not hit:
                continue
            item = {"fact": f["key"], "present": True,
                    "current": True, "certain": True,
                    "value": None}
            if self.behaviour != "naive":
                clause = ""
                for line in note.split("\n"):
                    for c in re.split(r"[.;]", line):
                        if hit.lower() in c.lower():
                            clause = c.lower()
                            break
                    if clause:
                        break
                if any(n in clause for n in
                       ("no ", "denies ", "not present", "is not")):
                    item["present"] = False
                if any(h in clause for h in
                       ("possible", "cannot rule out",
                        "suspected")):
                    item["certain"] = False
                if any(p in clause for p in
                       ("history of", "prior ", "in the past",
                        "resolved", "per prior note")):
                    item["current"] = False
            out.append(item)
        return json.dumps(out)

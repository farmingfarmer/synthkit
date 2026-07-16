"""SYNTH_V1 S3: corpus I/O — a run is a reproducible, auditable
object you can hand to a colleague.

Layout of corpus/<run_id>/:
    manifest.json    spec (verbatim), master seed, backend name,
                     render stats, sha256 of every artifact
    docs/<doc_id>.txt
    truth/<doc_id>.json      the blueprint — labels precede data
    render_report.json

load_corpus verifies every hash on the way in: a corpus that fails
integrity refuses to load, because an edited document silently
diverging from its blueprint would poison every evaluation after it.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .planner import Blueprint
from .renderer import RenderReport
from .spec import DataSpec


class CorpusIntegrityError(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_corpus(run_dir: Path, spec: DataSpec,
                 blueprints: List[Blueprint],
                 documents: Dict[str, str],
                 report: RenderReport,
                 backend_name: str = "unknown") -> Path:
    run_dir = Path(run_dir)
    (run_dir / "docs").mkdir(parents=True, exist_ok=True)
    (run_dir / "truth").mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    for bp in blueprints:
        text = documents.get(bp.doc_id, "")
        doc_rel = "docs/{}.txt".format(bp.doc_id)
        truth_rel = "truth/{}.json".format(bp.doc_id)
        (run_dir / doc_rel).write_text(text, encoding="utf-8")
        truth_json = bp.to_json()
        (run_dir / truth_rel).write_text(truth_json, encoding="utf-8")
        hashes[doc_rel] = _sha(text)
        hashes[truth_rel] = _sha(truth_json)
    report_json = json.dumps(asdict(report), indent=2)
    (run_dir / "render_report.json").write_text(report_json,
                                                encoding="utf-8")
    manifest = {
        "synthkit_manifest": 1,
        "spec": json.loads(spec.to_json()),
        "master_seed": spec.corpus.master_seed,
        "backend": backend_name,
        "documents": len(blueprints),
        "render": {
            "first_try": report.first_try,
            "retried": report.retried,
            "fallbacks": report.fallbacks,
        },
        "hashes": hashes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return run_dir


def load_corpus(run_dir: Path,
                ) -> Tuple[DataSpec, List[Blueprint],
                           Dict[str, str], Dict[str, Any]]:
    """Returns (spec, blueprints, documents, manifest). Every hash
    verified; any mismatch raises CorpusIntegrityError naming the
    file."""
    run_dir = Path(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = DataSpec.from_json(json.dumps(manifest["spec"]))
    documents: Dict[str, str] = {}
    blueprints: List[Blueprint] = []
    bad: List[str] = []
    for rel, expected in sorted(manifest["hashes"].items()):
        path = run_dir / rel
        text = path.read_text(encoding="utf-8")
        if _sha(text) != expected:
            bad.append(rel)
            continue
        if rel.startswith("docs/"):
            documents[path.stem] = text
        else:
            raw = json.loads(text)
            blueprints.append(_blueprint_from_raw(raw))
    if bad:
        raise CorpusIntegrityError(
            "corpus integrity failed for: {}".format(", ".join(bad)))
    blueprints.sort(key=lambda b: b.doc_index)
    return spec, blueprints, documents, manifest


def _blueprint_from_raw(raw: Dict[str, Any]) -> Blueprint:
    from .planner import (Blueprint, PlannedDistractor,
                          PlannedElement, PlannedNote)
    return Blueprint(
        doc_id=raw["doc_id"],
        doc_index=raw["doc_index"],
        seed=raw["seed"],
        structured=raw.get("structured", {}),
        notes=[
            PlannedNote(
                field_name=n["field_name"],
                note_type=n["note_type"],
                style=n["style"],
                length_words=n["length_words"],
                elements=[PlannedElement(**e)
                          for e in n.get("elements", [])],
                distractors=[PlannedDistractor(**d)
                             for d in n.get("distractors", [])],
            )
            for n in raw.get("notes", [])
        ],
    )

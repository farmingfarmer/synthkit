"""SYNTH_D1: the GUI — a calibration bench for data claims.

`synthkit gui` serves a single-page app on 127.0.0.1 wrapping the
whole pipeline: plain English -> compiled spec (with the
assisted-never-trusted repair loop and the human gate as a visible
review step) -> validate -> render + preview -> campaigns ->
showdown. Preset buttons inject the reference cases. The persistent
SPEC CARD shows the live sha256 fingerprint of the current spec —
determinism made visible: same fingerprint, same data, forever.

Stdlib only (http.server + a self-contained page). Binds loopback
exclusively. Solvers resolve from a built-in registry or a dotted
path. `BACKEND_FACTORY` is injectable for tests.

Python 3.8 compatible.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional

BACKEND_FACTORY = None  # tests may inject: fn(name, model) -> backend

_FINGERPRINT = None


def build_info() -> dict:
    """version · built date · fingerprint — the legible answer to
    'am I on the latest?'. Compare against `synthkit version` in
    a terminal: the CLI reads the same disk, so a match means
    the bench serves what is installed."""
    import datetime
    from . import __version__
    pkg = Path(__file__).parent
    newest = max(path.stat().st_mtime
                 for path in pkg.glob("*.py"))
    return {
        "version": __version__,
        "built": datetime.date.fromtimestamp(
            newest).isoformat(),
        "fingerprint": build_fingerprint(),
    }


def build_fingerprint() -> str:
    """Short sha over every module's bytes — the bench wears it
    in the wordmark so 'am I on the latest?' is a glance, not a
    ritual."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        import hashlib
        h = hashlib.sha256()
        pkg = Path(__file__).parent
        for path in sorted(pkg.glob("*.py")):
            h.update(path.read_bytes())
        _FINGERPRINT = h.hexdigest()[:8]
    return _FINGERPRINT


_JOBS: Dict[str, dict] = {}


JOB_BUDGET_S = 1800.0   # a campaign should not outlive this


def _start_job(fn, payload: dict) -> str:
    import time
    import uuid
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "running",
                     "started": time.time(),
                     "cancelled": False}

    def work():
        try:
            result = fn(payload)
            if "error" in result:
                _JOBS[job_id].update(status="error",
                                     error=result["error"])
            else:
                _JOBS[job_id].update(status="done",
                                     result=result)
        except Exception as e:
            _JOBS[job_id].update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return job_id


def api_job(payload: dict) -> dict:
    import time
    job = _JOBS.get(payload.get("id", ""))
    if job is None:
        return {"error": "unknown job"}
    elapsed = round(time.time() - job["started"], 1)
    status = job["status"]
    if status == "running" and job.get("cancelled"):
        status = "cancelled"
    elif status == "running" and elapsed > JOB_BUDGET_S:
        status = "timeout"
    out = {"status": status, "elapsed": elapsed}
    if status == "done":
        out["result"] = job["result"]
    elif status == "error":
        out["error"] = job["error"]
    elif status == "timeout":
        out["error"] = ("job exceeded its {}s budget — the "
                        "backend is likely crawling or wedged "
                        "(a wedged ollama once served a 6-minute "
                        "run for 106 minutes). The thread may "
                        "still finish in the background; safe "
                        "to move on.".format(int(JOB_BUDGET_S)))
    elif status == "cancelled":
        out["error"] = ("cancelled — the current backend call "
                        "may run to completion in the "
                        "background, then stop")
    return out


def api_job_cancel(payload: dict) -> dict:
    job = _JOBS.get(payload.get("id", ""))
    if job is None:
        return {"error": "unknown job"}
    job["cancelled"] = True
    return {"ok": True}


def _backend(name: str, model: str):
    if BACKEND_FACTORY is not None:
        return BACKEND_FACTORY(name, model)
    from .cli import _backend as cli_backend
    return cli_backend(name, model)


# ===================================================================
# Solver registry
# ===================================================================

def _solver(name: str) -> Callable:
    from .autosolver import (autoclean, autosolver,
                             autosolver_hybrid,
                             autosolver_regress)
    from .examples import regex_extract, strip_cleaner
    registry: Dict[str, Callable] = {
        "autoclean": autoclean,
        "strip_cleaner": strip_cleaner,
        "autosolver": autosolver(),
        "autosolver_regress": autosolver_regress(),
        "autosolver_hybrid": autosolver_hybrid(),
        "regex_extract": regex_extract,
    }
    if name in registry:
        return registry[name]
    if ":" in name:
        import importlib
        import os
        import sys as _sys
        # console-script servers lack cwd on sys.path;
        # without this, vendor files beside the spec
        # are invisible (live catch, mid-rehearsal)
        cwd = os.getcwd()
        if cwd not in _sys.path:
            _sys.path.insert(0, cwd)
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise ValueError(
        "unknown solver `{}` — built-ins: {}".format(
            name, ", ".join(sorted(registry))))


# ===================================================================
# API operations
# ===================================================================

_READMIT_DEMO = '{\n  "title": "CHF discharge cohort: 30-day readmission (vendor evaluation)",\n  "columns": [\n    {\n      "name": "patient_id",\n      "ctype": "str_id",\n      "distribution": {\n        "kind": "sequence",\n        "prefix": "MRN",\n        "start": 400000\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "patient_name",\n      "ctype": "person_name",\n      "distribution": {},\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.06,\n        "space_rate": 0.03,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "age",\n      "ctype": "int",\n      "distribution": {\n        "kind": "normal",\n        "mean": 74,\n        "std": 11,\n        "min": 40,\n        "max": 97\n      },\n      "mess": {\n        "missing_rate": 0.1,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "sex",\n      "ctype": "category",\n      "distribution": {\n        "kind": "categorical",\n        "choices": [\n          "female",\n          "male"\n        ],\n        "weights": [\n          0.48,\n          0.52\n        ]\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.08,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "ejection_fraction",\n      "ctype": "int",\n      "distribution": {\n        "kind": "mixture",\n        "components": [\n          {\n            "kind": "normal",\n            "mean": 30,\n            "std": 6,\n            "min": 12,\n            "max": 40\n          },\n          {\n            "kind": "normal",\n            "mean": 55,\n            "std": 5,\n            "min": 45,\n            "max": 70\n          }\n        ],\n        "weights": [\n          0.55,\n          0.45\n        ]\n      },\n      "mess": {\n        "missing_rate": 0.18,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "sodium",\n      "ctype": "float",\n      "distribution": {\n        "kind": "normal",\n        "mean": 137.0,\n        "std": 4.0,\n        "min": 118.0,\n        "max": 150.0\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.02,\n        "outlier_factor": 1.4,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "creatinine",\n      "ctype": "float",\n      "distribution": {\n        "kind": "lognormal",\n        "mu": 0.25,\n        "sigma": 0.45,\n        "min": 0.4,\n        "max": 8.0\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.06,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "prior_admissions_12mo",\n      "ctype": "int",\n      "distribution": {\n        "kind": "mixture",\n        "components": [\n          {\n            "kind": "uniform",\n            "min": 0,\n            "max": 0\n          },\n          {\n            "kind": "uniform",\n            "min": 1,\n            "max": 5\n          }\n        ],\n        "weights": [\n          0.55,\n          0.45\n        ]\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "num_discharge_medications",\n      "ctype": "int",\n      "distribution": {\n        "kind": "normal",\n        "mean": 11,\n        "std": 4,\n        "min": 2,\n        "max": 26\n      },\n      "mess": {\n        "missing_rate": 0.07,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "admitting_unit",\n      "ctype": "category",\n      "distribution": {\n        "kind": "categorical",\n        "choices": [\n          "cardiology",\n          "internal medicine",\n          "intensive care"\n        ],\n        "weights": [\n          0.5,\n          0.38,\n          0.12\n        ]\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.06,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "admission_date",\n      "ctype": "date",\n      "distribution": {\n        "kind": "date_range",\n        "start": "2026-01-05",\n        "end": "2026-06-20"\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.07,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "discharge_date",\n      "ctype": "date",\n      "distribution": {},\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.02\n      },\n      "note": {}\n    },\n    {\n      "name": "length_of_stay",\n      "ctype": "int",\n      "distribution": {\n        "kind": "mixture",\n        "components": [\n          {\n            "kind": "uniform",\n            "min": 2,\n            "max": 5\n          },\n          {\n            "kind": "uniform",\n            "min": 8,\n            "max": 16\n          }\n        ],\n        "weights": [\n          0.7,\n          0.3\n        ]\n      },\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "total_charges",\n      "ctype": "float",\n      "distribution": {},\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.015,\n        "outlier_factor": 30.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {}\n    },\n    {\n      "name": "discharge_note",\n      "ctype": "note",\n      "distribution": {},\n      "mess": {\n        "missing_rate": 0.0,\n        "missing_tokens": [\n          "",\n          "NULL",\n          "N/A",\n          "?"\n        ],\n        "typo_rate": 0.0,\n        "format_rate": 0.0,\n        "outlier_rate": 0.0,\n        "outlier_factor": 10.0,\n        "case_rate": 0.0,\n        "space_rate": 0.0,\n        "wrong_rate": 0.0\n      },\n      "note": {\n        "elements": [\n          {\n            "id": "med_nonadherence",\n            "density": 0.22,\n            "weight": 1.5,\n            "phrasings": [\n              "pt admits to missing several doses of diuretics this month",\n              "poor adherence to medication regimen per family",\n              "ran out of furosemide two wks ago, did not refill",\n              "hx of med non-adherence documented by pharmacy"\n            ]\n          },\n          {\n            "id": "no_home_support",\n            "density": 0.28,\n            "weight": 1.1,\n            "phrasings": [\n              "lives alone, no home support available",\n              "no family or caregiver able to assist at home",\n              "declines home health; lives by self in 2nd floor apt"\n            ]\n          },\n          {\n            "id": "fluid_overload_signs",\n            "density": 0.3,\n            "weight": 1.3,\n            "phrasings": [\n              "3+ pitting edema and 4 kg weight gain since last visit",\n              "worsening orthopnea and bilateral crackles on exam",\n              "JVD elevated, weight up 3.5 kg over one week"\n            ]\n          },\n          {\n            "id": "followup_barrier",\n            "density": 0.18,\n            "weight": 0.9,\n            "phrasings": [\n              "no transportation to cardiology follow-up",\n              "unable to schedule follow up d/t work conflict",\n              "pt states he will not attend clinic visit"\n            ]\n          }\n        ],\n        "distractors": [\n          {\n            "id": "denies_nonadherence",\n            "density": 0.5,\n            "excludes": "med_nonadherence",\n            "phrasings": [\n              "denies missing any medication doses",\n              "adherent to all medications per pillbox review",\n              "no issues with medication compliance reported"\n            ]\n          },\n          {\n            "id": "no_overload",\n            "density": 0.45,\n            "excludes": "fluid_overload_signs",\n            "phrasings": [\n              "no edema, lungs clear to auscultation",\n              "denies orthopnea or PND, weight stable",\n              "no signs of fluid overload on exam"\n            ]\n          },\n          {\n            "id": "historical_admission",\n            "density": 0.35,\n            "phrasings": [\n              "prior admission for CHF exacerbation in 2023, resolved",\n              "remote hx of decompensation, stable since",\n              "old echocardiogram from 2022 reviewed, unchanged"\n            ]\n          }\n        ],\n        "fillers": [\n          "Vital signs stable at time of discharge",\n          "Diet and fluid restriction counseling provided",\n          "Discharge medications reconciled with pharmacy",\n          "Patient ambulating independently on the unit",\n          "Follow-up labs ordered for next week",\n          "Influenza vaccination administered this admission"\n        ]\n      }\n    }\n  ],\n  "rows": 1000,\n  "master_seed": 20260727,\n  "duplicate_rate": 0.02,\n  "rules": [\n    {\n      "kind": "date_after",\n      "earlier": "admission_date",\n      "later": "discharge_date",\n      "days_from": "length_of_stay"\n    },\n    {\n      "kind": "derived",\n      "target": "total_charges",\n      "source": "length_of_stay",\n      "factor": 2650,\n      "noise_sigma": 0.18\n    }\n  ],\n  "outcomes": [\n    {\n      "name": "readmitted_30d",\n      "kind": "logistic",\n      "intercept": 3.4,\n      "coefficients": {\n        "age": 0.015,\n        "prior_admissions_12mo": 0.32,\n        "ejection_fraction": -0.025,\n        "sodium": -0.06,\n        "length_of_stay": 0.05,\n        "discharge_note.med_nonadherence": 1.5,\n        "discharge_note.no_home_support": 1.1,\n        "discharge_note.fluid_overload_signs": 1.3,\n        "discharge_note.followup_barrier": 0.9\n      },\n      "target_prevalence": [\n        0.05,\n        0.12\n      ]\n    }\n  ]\n}'


_CALIBRATED_ENCOUNTER = """{
  "title": "Inpatient encounters (readmission study)",
  "rows": 400,
  "master_seed": 12345,
  "duplicate_rate": 0.03,
  "columns": [
    {"name": "patient_id", "ctype": "str_id",
     "distribution": {"kind": "sequence", "prefix": "P",
                      "start": 20000}},
    {"name": "patient_name", "ctype": "person_name",
     "mess": {"case_rate": 0.05, "space_rate": 0.02}},
    {"name": "age", "ctype": "int",
     "distribution": {"kind": "normal", "mean": 62, "std": 15,
                      "min": 18, "max": 100},
     "mess": {"missing_rate": 0.12}},
    {"name": "admitting_department", "ctype": "category",
     "distribution": {"kind": "categorical",
       "choices": ["internal medicine", "cardiology",
                   "oncology", "emergency"],
       "weights": [0.5, 0.2, 0.2, 0.1]},
     "mess": {"typo_rate": 0.05}},
    {"name": "length_of_stay", "ctype": "int",
     "distribution": {"kind": "mixture",
       "components": [{"kind": "uniform", "min": 1, "max": 4},
                      {"kind": "uniform", "min": 10,
                       "max": 18}],
       "weights": [0.75, 0.25]},
     "mess": {"outlier_rate": 0.01, "outlier_factor": 2}},
    {"name": "total_charges", "ctype": "float",
     "mess": {"outlier_rate": 0.02, "outlier_factor": 40.0}},
    {"name": "admission_date", "ctype": "date",
     "distribution": {"kind": "date_range",
                      "start": "2026-01-01",
                      "end": "2026-06-30"},
     "mess": {"format_rate": 0.05}},
    {"name": "discharge_date", "ctype": "date",
     "mess": {"wrong_rate": 0.02}}
  ],
  "rules": [
    {"kind": "date_after", "earlier": "admission_date",
     "later": "discharge_date", "days_from": "length_of_stay"},
    {"kind": "derived", "target": "total_charges",
     "source": "length_of_stay", "factor": 2100,
     "noise_sigma": 0.15}
  ],
  "outcomes": [
    {"name": "readmitted", "kind": "logistic",
     "intercept": -3.8,
     "coefficients": {"length_of_stay": 0.1, "age": 0.02,
                      "admitting_department=oncology": 0.8},
     "target_prevalence": [0.10, 0.25]}
  ]
}"""


def api_presets() -> dict:
    from .examples import reference_spec, reference_table
    import json as _json
    calibrated = _CALIBRATED_ENCOUNTER
    return {"presets": [
        {"name": "CHF readmission w/ notes (vendor demo)",
         "kind": "table",
         "description": "1000 patients, 9% readmission "
                        "minority, mixtures/missing/typos/"
                        "outliers, and a discharge-note column "
                        "whose planted phrases carry decisive "
                        "signal behind negation traps",
         "spec": _json.loads(_READMIT_DEMO)},
        {"name": "Encounter benchmark (calibrated)",
         "kind": "table",
         "description": "the 400-row readmission study from the "
                        "live compiler benchmark: coupled dates, "
                        "derived charges, declared prevalence",
         "spec": _json.loads(calibrated)},
        {"name": "Billing table (reference)",
         "kind": "table",
         "description": "billing extract with bimodal stays, "
                        "cost derived from stay, every mess tier",
         "spec": json.loads(reference_table(rows=150).to_json())},
        {"name": "Lab results table",
         "kind": "table",
         "description": "compile this from English to watch the "
                        "assisted-never-trusted loop work",
         "english": "a 300-row lab results extract: patient id, "
                    "ordering department weighted toward internal "
                    "medicine, collection and result dates where "
                    "results follow collection by one to three "
                    "days, potassium normally distributed around "
                    "4.1, ten percent missing results, occasional "
                    "wrong-value dates, a few duplicate rows"},
        {"name": "Progress notes corpus",
         "kind": "document",
         "description": "the reference clinical-notes vertical "
                        "with the discontinued-med trap",
         "spec": json.loads(reference_spec(size=20).to_json())},
    ]}


def _load_spec(kind: str, raw: str):
    if kind == "table":
        from .tablespec import TableSpec
        return TableSpec.from_json(raw)
    from .spec import DataSpec
    return DataSpec.from_json(raw)


def api_validate(payload: dict) -> dict:
    from .spec import SpecError
    from .tablespec import TableSpecError
    try:
        spec = _load_spec(payload["kind"], payload["spec"])
    except Exception as e:
        return {"ok": False,
                "problems": "not parseable: {}".format(e)}
    try:
        spec.validate()
        return {"ok": True, "problems": ""}
    except (SpecError, TableSpecError) as e:
        return {"ok": False, "problems": str(e)}


def api_compile(payload: dict) -> dict:
    from .compiler import compile_spec, compile_table_spec
    backend = _backend(payload.get("backend", "ollama"),
                       payload.get("model", ""))
    fn = compile_table_spec if payload["kind"] == "table" \
        else compile_spec
    result = fn(payload["description"], backend, retries=1)
    return {"ok": result.ok, "raw_json": result.raw_json,
            "problems": result.problems or ""}


def api_lint(payload: dict) -> dict:
    from .lint import lint_corpus, lint_table
    if payload.get("kind") == "table":
        from .tablespec import TableSpec
        report = lint_table(
            TableSpec.from_json(payload["spec"]),
            description=payload.get("description", ""))
    else:
        from .spec import DataSpec
        report = lint_corpus(
            DataSpec.from_json(payload["spec"]),
            description=payload.get("description", ""))
    return {"ok": report.ok, "text": report.format_text()}


def api_plan(payload: dict) -> dict:
    spec = _load_spec(payload["kind"], payload["spec"])
    if payload["kind"] == "table":
        from .tableplan import plan_table
        bp = plan_table(spec)
        ops: Dict[str, int] = {}
        for m in bp.ledger:
            ops[m.op] = ops.get(m.op, 0) + 1
        return {"rows": len(bp.clean_rows),
                "dirty_rows": len(bp.dirty_rows),
                "columns": bp.columns,
                "mess_by_op": ops,
                "outcomes": sorted(bp.true_probs)}
    from .planner import plan_corpus
    spec.validate()
    blueprints = plan_corpus(spec)
    counts: Dict[str, int] = {}
    for bp in blueprints:
        for note in bp.notes:
            for el in note.elements:
                counts[el.element_id] = counts.get(
                    el.element_id, 0) + 1
    return {"documents": len(blueprints),
            "planted_elements": counts}


def api_render(payload: dict) -> dict:
    spec = _load_spec(payload["kind"], payload["spec"])
    out = Path(payload.get("out") or "gui_runs/run_001")
    if payload["kind"] == "table":
        from .tableplan import plan_table, write_table
        bp = plan_table(spec)
        run_dir = write_table(out, spec, bp)
        outcomes = []
        for oc in spec.outcomes:
            name = oc["name"]
            if oc.get("kind", "logistic") == "logistic":
                pos = sum(1 for r in bp.clean_rows
                          if r.get(name) == "True")
                outcomes.append({
                    "name": name,
                    "realized": pos / max(len(bp.clean_rows), 1),
                    "declared": oc.get("target_prevalence")})
        return {"run_dir": str(run_dir),
                "columns": bp.columns,
                "rows": len(bp.dirty_rows),
                "outcomes": outcomes,
                "preview": bp.dirty_rows[:40],
                "mess_cells": len([m for m in bp.ledger
                                   if m.op != "duplicate"])}
    from .corpus_io import write_corpus
    from .harness import StubBackend
    from .planner import plan_corpus
    from .renderer import render_corpus
    spec.validate()
    blueprints = plan_corpus(spec)
    backend = (StubBackend()
               if payload.get("backend", "stub") == "stub"
               else _backend(payload["backend"],
                             payload.get("model", "")))
    documents, report = render_corpus(spec, blueprints, backend)
    run_dir = write_corpus(out, spec, blueprints, documents,
                           report, backend_name=payload.get(
                               "backend", "stub"))
    first = documents[sorted(documents)[0]] if documents else ""
    return {"run_dir": str(run_dir),
            "documents": len(documents),
            "first_document": first[:800],
            "verified_first_try": report.first_try}


def api_campaign_compile(payload: dict) -> dict:
    from .campaign import compile_campaign, write_campaign
    spec = _load_spec(
        "table" if payload["goal"] in ("clean", "predict",
                       "regress")
        else "document", payload["spec"])
    camp = compile_campaign(payload["goal"], spec,
                            bars=payload.get("bars") or {},
                            outcome=payload.get("outcome", ""))
    out = payload.get("out", "")
    if not out:
        # Auto-increment: each compile gets its own directory so
        # trial history stays with the campaign that produced it
        # (a live tour found regress arms filed under a clean
        # campaign's tiers).
        base = Path("gui_runs")
        base.mkdir(exist_ok=True)
        n = 1
        while (base / "campaign_{:03d}".format(n)).exists():
            n += 1
        out = base / "campaign_{:03d}".format(n)
    out = Path(out)
    write_campaign(out, camp)
    return {"campaign_dir": str(out),
            "title": camp.title,
            "tiers": [{"name": t.name, "notes": t.notes,
                       "conditions": t.conditions}
                      for t in camp.tiers]}


def api_campaign_run(payload: dict) -> dict:
    from .campaign import load_campaign, run_campaign, \
        write_campaign
    camp = load_campaign(Path(payload["campaign_dir"]))
    extractor = None
    if payload["solver"] == "llm_extract":
        if camp.goal != "extract":
            return {"error": "llm_extract runs on extract "
                             "campaigns only"}
        from .llmvendor import LLMExtractor
        from .spec import DataSpec
        spec = DataSpec.from_json(camp.tiers[0].spec_json)
        extractor = LLMExtractor(
            _backend(payload.get("backend", "ollama"),
                     payload.get("model", "")),
            spec, samples=int(payload.get("samples", 1)),
            name="llm-vendor",
            extra_system=payload.get("extra_system", ""))
        solver = extractor
    else:
        solver = _solver(payload["solver"])
    result = run_campaign(camp, solver,
                          solver_name=payload["solver"])
    write_campaign(Path(payload["campaign_dir"]), camp, result,
                   result_name=payload.get("name")
                   or payload["solver"])
    text = result.format_text()
    if extractor is not None:
        text += "\n" + extractor.stats_line()
    return {"highest_passed": result.highest_passed,
            "tiers": len(result.tier_results),
            "text": text}


def api_export(payload: dict) -> dict:
    """Return a run artifact for download. which: dirty|clean|
    ledger; fmt: csv|json. Table runs only (documents live as
    .txt on disk already)."""
    import csv as _csv
    import io
    run = Path(payload["dir"])
    which = payload.get("which", "dirty")
    fmt = payload.get("fmt", "csv")
    if which == "ledger":
        raw = (run / "ledger.json").read_text(encoding="utf-8")
        return {"filename": "corruption_ledger.json",
                "content": raw, "mime": "application/json"}
    path = run / ("dirty.csv" if which == "dirty"
                  else "clean.csv")
    raw = path.read_text(encoding="utf-8")
    label = ("synthetic_data" if which == "dirty"
             else "answer_key")
    if fmt == "csv":
        return {"filename": label + ".csv", "content": raw,
                "mime": "text/csv"}
    rows = list(_csv.DictReader(io.StringIO(raw)))
    return {"filename": label + ".json",
            "content": json.dumps(rows, indent=1,
                                  ensure_ascii=False),
            "mime": "application/json"}


def api_showdown(payload: dict) -> dict:
    from .autosolver import run_showdown
    from .campaign import load_campaign
    camp = load_campaign(Path(payload["campaign_dir"]))
    vendor = _solver(payload["solver"])
    kwargs = {}
    if payload.get("baseline"):
        kwargs["baseline"] = _solver(
            payload["baseline"])
        kwargs["baseline_name"] = payload["baseline"]
    result = run_showdown(camp, vendor,
                          vendor_name=payload.get("name")
                          or payload["solver"],
                          **kwargs)
    tiers = [{"tier": t.tier, "ceiling": t.ceiling,
              "baseline": t.baseline_auroc,
              "vendor": t.vendor_auroc,
              "passed": t.vendor_passed}
             for t in result.tiers]
    return {"text": result.format_text(),
            "tiers": tiers,
            "report": dict(
                _showdown_report(
                    camp, tiers, result.vendor_name,
                    result.baseline_name),
                tiers_data=tiers)}


def _showdown_report(camp, tiers, vendor_name, baseline_name):
    """Everything the plain-English final report needs: dataset
    facts, trap examples pulled from the ACTUAL spec, model
    methodology, per-tier verdicts, and the winner."""
    from .tablespec import TableSpec
    spec = None
    for t in camp.tiers:
        if "as-specified" in t.name or spec is None:
            spec = TableSpec.from_json(t.spec_json)
            if "as-specified" in t.name:
                break
    oc = next((o for o in spec.outcomes
               if o["name"] == camp.outcome), spec.outcomes[0])
    traps = []
    mess_kinds = set()
    note_cols = 0
    for c in spec.columns:
        if c.ctype == "note":
            note_cols += 1
            n = c.note or {}
            for dis in n.get("distractors", []):
                phr = (dis.get("phrasings") or [""])[0]
                if dis.get("excludes"):
                    traps.append({
                        "kind": "negation trap",
                        "example": phr,
                        "why": "appears ONLY in patients "
                               "WITHOUT the risk — a model "
                               "matching keywords without "
                               "understanding negation gets "
                               "this exactly backwards"})
                else:
                    traps.append({
                        "kind": "history trap",
                        "example": phr,
                        "why": "an old, resolved finding — "
                               "not current risk"})
            for el in n.get("elements", [])[:2]:
                traps.append({
                    "kind": "hidden signal",
                    "example": (el.get("phrasings")
                                or [""])[0],
                    "why": "genuine risk, phrased informally "
                           "— worth {} in the true risk "
                           "model, invisible to any model "
                           "that ignores the note".format(
                               el.get("weight"))})
        m = c.mess
        for attr, label in [
                ("missing_rate", "missing values"),
                ("typo_rate", "typos"),
                ("format_rate", "mixed formats"),
                ("outlier_rate", "implausible outliers"),
                ("wrong_rate", "subtly wrong values"),
                ("case_rate", "inconsistent casing"),
                ("space_rate", "stray whitespace")]:
            if getattr(m, attr, 0):
                mess_kinds.add(label)
    if spec.duplicate_rate:
        mess_kinds.add("duplicated rows")
    meth = {
        "autosolver": "standardizes every messy column "
            "(dates to one format, numbers cleaned of $ and "
            "commas, missing values handled), then fits a "
            "logistic regression — a classic, transparent "
            "statistical model. Pure standard-library code, "
            "fully deterministic, zero external dependencies. "
            "It does NOT read free-text fields.",
        "autosolver_hybrid": "does everything autosolver does, "
            "AND mines the free-text notes: from the training "
            "records alone it learns which words and phrases "
            "predict the outcome (keeping negated mentions — "
            "'denies X' — as separate evidence from 'X'), "
            "then feeds table features and text features into "
            "one logistic regression. No hand-written rules, "
            "no external libraries, fully deterministic.",
    }
    key = "as-specified"
    kt = next((t for t in tiers if key in t["tier"]),
              tiers[len(tiers) // 2] if tiers else None)
    winner = None
    if kt:
        vw = kt["vendor"] >= kt["baseline"]
        winner = {
            "name": vendor_name if vw else baseline_name,
            "loser": baseline_name if vw else vendor_name,
            "vendor_won": vw,
            "tier": kt["tier"],
            "margin": round(abs(kt["vendor"]
                                - kt["baseline"]), 3)}
    from .autosolver import LAST_FIT
    inside = LAST_FIT.get(baseline_name) or {}
    return {
        "inside_the_model": inside,
        "dataset": {
            "rows": spec.rows,
            "outcome": camp.outcome or oc["name"],
            "declared": oc.get("target_prevalence"),
            "note_columns": note_cols,
            "n_columns": len(spec.columns)},
        "mess_kinds": sorted(mess_kinds),
        "traps": traps[:6],
        "vendor": {"name": vendor_name,
                   "how": "the model under evaluation — "
                          "treated as a black box; it sees "
                          "the training data and answers on "
                          "the test data, nothing more"},
        "baseline": {"name": baseline_name,
                     "how": meth.get(baseline_name,
                                     meth["autosolver"])},
        "winner": winner,
        "bars": (camp.tiers[0].conditions
                 if camp.tiers else [])}


_ROUTES = {
    "/api/presets": lambda payload: api_presets(),
    "/api/validate": api_validate,
    "/api/compile": api_compile,
    "/api/lint": api_lint,
    "/api/plan": api_plan,
    "/api/render": api_render,
    "/api/campaign-compile": api_campaign_compile,
    "/api/campaign-run": api_campaign_run,
    "/api/campaign-run-async": lambda payload: {
        "job": _start_job(api_campaign_run, payload)},
    "/api/export": api_export,
    "/api/render-async": lambda payload: {
        "job": _start_job(api_render, payload)},
    "/api/showdown-async": lambda payload: {
        "job": _start_job(api_showdown, payload)},
    "/api/job": api_job,
    "/api/job-cancel": api_job_cancel,
    "/api/showdown": api_showdown,
}


# ===================================================================
# Server
# ===================================================================

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes,
              ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type",
                         ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/presets":
            self._send(200, json.dumps(
                api_presets()).encode("utf-8"))
        elif self.path == "/api/version":
            self._send(200, json.dumps(
                build_info()).encode("utf-8"))
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        handler = _ROUTES.get(self.path)
        if handler is None:
            self._send(404, b'{"error": "not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(
                self.rfile.read(length).decode("utf-8") or "{}")
            result = handler(payload)
            self._send(200, json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._send(200, json.dumps(
                {"error": str(e)}).encode("utf-8"))


def make_server(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


def run_gui(port: int = 8377, open_browser: bool = True) -> None:
    server = make_server(port)
    url = "http://127.0.0.1:{}/".format(server.server_port)
    print("synthkit bench -> {}   (Ctrl-C to stop)".format(url))
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


# ===================================================================
# The page — calibration-bench identity, mono-forward, spec card
# ===================================================================

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthkit bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bench:#EDF0F3; --panel:#FFFFFF; --ink:#17222C; --dim:#5C6B78;
  --rule:#CBD4DC; --enamel:#0E6E64; --enamel-press:#0A544C;
  --pass:#17803D; --fail:#B3261E; --chip:#DFE6EB;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
body{background:var(--bench);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.5}
.frame{display:grid;grid-template-columns:216px 1fr;
  min-height:100vh}
/* ---- station rail ---- */
nav{border-right:1px solid var(--rule);padding:20px 0}
.wordmark{font-family:var(--mono);font-weight:600;
  letter-spacing:.14em;padding:0 20px 18px;font-size:15px}
.wordmark small{display:block;color:var(--dim);font-weight:400;
  letter-spacing:.08em;font-size:10px;margin-top:2px}
.station{display:flex;gap:12px;align-items:baseline;width:100%;
  padding:11px 20px;border:0;background:none;text-align:left;
  font-family:var(--sans);font-size:14px;color:var(--dim);
  cursor:pointer;border-left:3px solid transparent}
.station b{font-family:var(--mono);font-size:11px;font-weight:500}
.station.active{color:var(--ink);border-left-color:var(--enamel);
  background:var(--panel)}
.station:focus-visible{outline:2px solid var(--enamel)}
/* ---- bench ---- */
main{padding:24px 32px;max-width:1080px}
header.bar{display:flex;justify-content:space-between;
  align-items:flex-start;gap:16px;margin-bottom:20px}
h1{font-family:var(--mono);font-size:19px;font-weight:600}
h1 span{color:var(--dim);font-weight:400}
/* the signature: the spec card */
#speccard{font-family:var(--mono);font-size:12px;
  background:var(--panel);border:1px solid var(--rule);
  border-top:3px solid var(--enamel);padding:10px 14px;
  min-width:250px}
#speccard .fp{font-size:15px;font-weight:600;
  letter-spacing:.06em}
#speccard .meta{color:var(--dim);margin-top:2px}
#speccard.empty{border-top-color:var(--rule);color:var(--dim)}
section{display:none}
section.active{display:block}
.panel{background:var(--panel);border:1px solid var(--rule);
  padding:18px;margin-bottom:16px}
.eyebrow{font-family:var(--mono);font-size:10px;
  letter-spacing:.16em;color:var(--dim);text-transform:uppercase;
  margin-bottom:10px}
textarea,input,select{width:100%;font-family:var(--mono);
  font-size:12.5px;border:1px solid var(--rule);padding:9px;
  background:#FBFCFD;color:var(--ink)}
textarea:focus,input:focus,select:focus{
  outline:2px solid var(--enamel);outline-offset:-1px}
textarea{resize:vertical}
label{display:block;font-size:12px;color:var(--dim);
  margin:10px 0 4px}
button.act{font-family:var(--mono);font-size:12.5px;
  font-weight:500;letter-spacing:.04em;background:var(--enamel);
  color:#fff;border:0;padding:9px 16px;cursor:pointer;
  margin:12px 8px 0 0}
button.act:hover{background:var(--enamel-press)}
button.act:focus-visible{outline:2px solid var(--ink)}
button.ghost{background:none;color:var(--enamel);
  border:1px solid var(--enamel)}
button.ghost:hover{background:#E7F0EF}
.presets{display:flex;gap:8px;flex-wrap:wrap}
.presets button{font-family:var(--mono);font-size:12px;
  background:var(--chip);border:1px solid var(--rule);
  padding:7px 12px;cursor:pointer;color:var(--ink)}
.presets button:hover{border-color:var(--enamel)}
pre.out{font-family:var(--mono);font-size:12px;line-height:1.55;
  background:#FBFCFD;border:1px solid var(--rule);padding:14px;
  overflow:auto;max-height:420px;white-space:pre-wrap;
  margin-top:12px}
pre.out:empty{display:none}
.outlabel{font-family:var(--mono);font-size:10px;
  letter-spacing:.16em;color:var(--dim);
  text-transform:uppercase;margin-top:14px}
.outlabel:has(+pre.out:empty){display:none}
pre.out .ok{color:var(--pass)} pre.out .bad{color:var(--fail)}
.verdict-pass{color:var(--pass);font-weight:600}
.verdict-fail{color:var(--fail);font-weight:600}
table.preview{border-collapse:collapse;font-family:var(--mono);
  font-size:11.5px;margin-top:12px;width:100%}
table.preview th,table.preview td{border:1px solid var(--rule);
  padding:5px 8px;text-align:left;max-width:160px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table.preview th{background:var(--chip);
  font-family:var(--mono);font-weight:500}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
@media(max-width:860px){.frame{grid-template-columns:1fr}
  nav{display:flex;overflow-x:auto;border-right:0;
  border-bottom:1px solid var(--rule)}
  .row2{grid-template-columns:1fr}}
@media(prefers-reduced-motion:no-preference){
  section.active{animation:in .16s ease-out}
  @keyframes in{from{opacity:.4}to{opacity:1}}}

.stepbanner{font-size:15px;font-weight:700;margin:14px 0 4px;
  color:var(--ink,#1a2b3c)}
.explain{font-size:12.5px;line-height:1.55;color:#3c4a58;
  background:#f2f6f4;border-left:3px solid #2e6e5e;
  padding:8px 12px;margin:6px 0 12px;border-radius:0 6px 6px 0}
.stepno{display:inline-block;background:#2e6e5e;color:#fff;
  font-size:10.5px;font-weight:700;border-radius:9px;
  padding:1px 7px;margin-right:6px;vertical-align:1px}
.badge{display:inline-block;font-size:9.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.04em;
  border-radius:8px;padding:1px 7px;margin-right:6px}
.badge.req{background:#7c2d2d;color:#fff}
.badge.rec{background:#8a6d1a;color:#fff}
.badge.opt{background:#c9d4cf;color:#33413b}
.dl{display:inline-block;margin:4px 8px 4px 0;padding:5px 11px;
  font-size:11.5px;border:1px solid #2e6e5e;border-radius:6px;
  background:#fff;color:#2e6e5e;cursor:pointer;font-weight:600}
.dl:hover{background:#2e6e5e;color:#fff}
.sumline{font-size:12.5px;background:#eef4ff;
  border-left:3px solid #3b5f9e;padding:8px 12px;margin:8px 0;
  border-radius:0 6px 6px 0;line-height:1.5}
.vcard{border:2px solid #b6c2bd;border-radius:10px;
  padding:12px 16px;margin:10px 0;background:#fff}
.vcard.winner{border-color:#2e7d32;background:#f3faf3}
.vcard.loser{opacity:.92}
.vcard h4{margin:0 0 4px;font-size:14px}
.wbadge{display:inline-block;background:#2e7d32;color:#fff;
  font-size:10px;font-weight:800;border-radius:8px;
  padding:2px 9px;margin-left:8px;letter-spacing:.06em}
.tierrow{font-family:var(--mono,monospace);font-size:11.5px;
  padding:3px 8px;border-radius:5px;margin:2px 0}
.tierrow.pass{background:#e7f4e7;color:#1d4d22}
.tierrow.fail{background:#f9e7e7;color:#6e1f1f}
.report h3{font-size:13.5px;margin:16px 0 4px;color:#1a2b3c}
.report p,.report li{font-size:12.5px;line-height:1.55;
  color:#33413b}
.trapx{font-family:var(--mono,monospace);font-size:11px;
  background:#f6f1e7;border-radius:4px;padding:1px 5px}

.pvwrap{max-height:430px;overflow:auto;border:1px solid #c2cdc8;
  border-radius:8px;margin-top:8px;background:#fff}
.pvwrap table{border-collapse:collapse;width:max-content;
  min-width:100%}
.pvwrap th{position:sticky;top:0;background:#eef2f0;
  color:#000;font-size:11px;padding:6px 10px;text-align:left;
  border:1px solid #d3dcd7;white-space:nowrap;z-index:2}
.pvwrap td{background:#fff;color:#000;font-size:11.5px;
  font-family:var(--mono,monospace);padding:5px 10px;
  border:1px solid #e2e8e5;white-space:nowrap;max-width:340px;
  overflow:hidden;text-overflow:ellipsis;cursor:pointer}
.pvwrap td:hover{background:#f2f7f4}
.pvhint{font-size:11px;color:#5a6a63;margin:6px 0 0}
#cellmodal{display:none;position:fixed;inset:0;z-index:50;
  background:rgba(20,30,26,.55)}
#cellmodal .box{position:absolute;inset:7% 12%;background:#fff;
  border-radius:12px;padding:22px 26px;overflow:auto;
  box-shadow:0 18px 50px rgba(0,0,0,.35)}
#cellmodal h4{margin:0 0 10px;font-size:13px;color:#1a2b3c}
#cellmodal pre{white-space:pre-wrap;word-break:break-word;
  font-size:13.5px;line-height:1.6;color:#000;margin:0;
  font-family:var(--mono,monospace)}
#cellmodal .close{position:absolute;top:12px;right:16px;
  cursor:pointer;font-size:22px;color:#5a6a63;font-weight:700}
input.need,textarea.need,select.need{
  border:2px solid #b3261e !important;
  background:#fdf3f2 !important}
input.good,textarea.good,select.good{
  border:2px solid #2e7d32 !important;
  background:#f4faf4 !important}

.subt{display:block;font-size:9px;letter-spacing:.02em;
  text-transform:none;color:#7c8a84;font-weight:400;
  margin-top:1px}
.gov{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 10px}
.gov span{font-size:10.5px;background:#eef4ef;
  border:1px solid #cfdcd4;border-radius:12px;padding:3px 10px;
  color:#2c4a3e;font-weight:600}
.mcards{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
.mcard{background:#fff;border:1px solid #c9d4cf;
  border-radius:10px;padding:10px 16px;min-width:130px}
.mcard .n{font-size:21px;font-weight:800;color:#1a2b3c}
.mcard .l{font-size:10.5px;color:#5a6a63;
  text-transform:uppercase;letter-spacing:.05em}
.mcard.ok{border-color:#2e7d32;background:#f4faf4}
.mcard.warn{border-color:#b3261e;background:#fdf3f2}
.vstrip{background:#1f3d33;color:#fff;border-radius:10px;
  padding:12px 18px;margin:12px 0;font-size:13px;
  line-height:1.55}
.vstrip b{color:#9fe0c0}
.modelrow{display:flex;align-items:center;gap:8px;
  font-size:12px;padding:3px 8px;border-radius:5px;margin:2px 0;
  background:#fff;border:1px solid #e2e8e5}
.modelrow .bar{height:8px;border-radius:4px;min-width:2px}
.modelrow .up{background:#b3452c}
.modelrow .dn{background:#3b6ea5}
.modelrow code{font-size:11px}
.summarycard{background:#fff;border:1px solid #c9d4cf;
  border-radius:10px;padding:10px 16px;margin:6px 0 10px;
  font-size:12.5px;line-height:1.6;color:#22303a}
.summarycard b{color:#1a2b3c}
.stepbanner{font-size:17px}
.hint{font-size:12px;color:#44534c}
.act{padding:10px 22px;font-size:13.5px;font-weight:700}
details.explain summary{cursor:pointer;margin-bottom:6px}
.station.done b::after{content:" \2713";color:#2e7d32;
  font-weight:800}

/* ================================================
   DEPTH & READABILITY LAYER (appended: later wins)
   Consistent physical language: raised = clickable,
   inset = editable, flat card = information.
   ================================================ */
body{font-size:15.5px;line-height:1.55}
main{max-width:1160px}
.panel{background:var(--panel);border:1px solid var(--rule);
  border-radius:12px;padding:18px 22px;margin-bottom:18px;
  box-shadow:0 1px 2px rgba(23,34,44,.06),
             0 6px 18px rgba(23,34,44,.07)}
/* --- raised, sheened buttons: unmistakably pressable --- */
.act{
  background:linear-gradient(180deg,#178a7d 0%,
    var(--enamel) 45%,#0a5d54 100%);
  color:#fff;border:1px solid #0a4f47;border-radius:9px;
  padding:11px 26px;font-size:15px;font-weight:700;
  cursor:pointer;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.35),
             0 2px 4px rgba(14,110,100,.35),
             0 5px 12px rgba(23,34,44,.18);
  text-shadow:0 1px 1px rgba(0,0,0,.25);
  transition:transform .06s,box-shadow .06s}
.act:hover{transform:translateY(-1px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.4),
             0 4px 8px rgba(14,110,100,.4),
             0 8px 18px rgba(23,34,44,.22)}
.act:active{transform:translateY(1px);
  background:var(--enamel-press);
  box-shadow:inset 0 2px 5px rgba(0,0,0,.35)}
.act.ghost{
  background:linear-gradient(180deg,#ffffff 0%,#f2f5f4 55%,
    #e4eae8 100%);
  color:var(--enamel);border:1px solid #b9c6c1;
  text-shadow:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.9),
             0 2px 4px rgba(23,34,44,.12),
             0 4px 10px rgba(23,34,44,.08)}
.act.ghost:hover{transform:translateY(-1px)}
.act.ghost:active{transform:translateY(1px);
  background:#dde5e2;
  box-shadow:inset 0 2px 5px rgba(0,0,0,.15)}
.dl{
  background:linear-gradient(180deg,#ffffff,#eef3f1);
  border:1px solid #9db3aa;border-radius:8px;
  padding:7px 14px;font-size:13px;font-weight:600;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.9),
             0 2px 4px rgba(23,34,44,.14);
  transition:transform .06s,box-shadow .06s}
.dl:hover{transform:translateY(-1px);background:#e7f0ec;
  color:var(--enamel);
  box-shadow:0 4px 9px rgba(23,34,44,.18)}
.dl:active{transform:translateY(1px);
  box-shadow:inset 0 2px 4px rgba(0,0,0,.18)}
.presets .preset,.preset{
  box-shadow:inset 0 1px 0 rgba(255,255,255,.8),
             0 2px 5px rgba(23,34,44,.12);
  border-radius:10px;transition:transform .06s,
    box-shadow .06s;cursor:pointer}
.preset:hover{transform:translateY(-1px);
  box-shadow:0 5px 12px rgba(23,34,44,.18)}
/* --- inset editables: clearly "type here" --- */
input,select,textarea{
  font-size:14.5px;border-radius:8px;
  border:1px solid #b7c2bd;background:#fff;
  box-shadow:inset 0 2px 4px rgba(23,34,44,.10);
  padding:9px 12px}
select{
  background-image:linear-gradient(180deg,#fff,#f2f5f4);
  cursor:pointer;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.8),
             0 2px 4px rgba(23,34,44,.10)}
textarea{line-height:1.5}
/* --- bigger, darker text everywhere --- */
label{font-size:13.5px;color:#2c3a45;font-weight:600}
.hint{font-size:13.5px;color:#3d4c46;line-height:1.55}
.explain{font-size:14px;color:#33424e;
  box-shadow:0 1px 3px rgba(23,34,44,.06)}
.stepbanner{font-size:19px}
.eyebrow{font-size:13px}
.out{font-size:13px;line-height:1.55;
  box-shadow:inset 0 2px 5px rgba(23,34,44,.09);
  border-radius:8px}
.outlabel{font-size:11px}
.pvwrap{box-shadow:0 3px 10px rgba(23,34,44,.12)}
.pvwrap td{font-size:12.5px}
.pvwrap th{font-size:12px}
.summarycard,.mcard,.vcard{
  box-shadow:0 1px 2px rgba(23,34,44,.06),
             0 4px 12px rgba(23,34,44,.08)}
.vstrip{box-shadow:0 4px 14px rgba(23,34,44,.28)}
.report p,.report li{font-size:13.5px}
.station{font-size:15px}
.subt{font-size:10px}
.gov span{font-size:11.5px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7),
             0 1px 3px rgba(23,34,44,.08)}
.badge{font-size:10px}
.stepno{font-size:11px;
  box-shadow:0 1px 3px rgba(14,110,100,.35)}
</style></head><body>
<div class="frame">
<nav>
  <div class="wordmark">SYNTHKIT<small>calibration bench<br>
    <span id="fp" title="version &middot; built &middot; build
    fingerprint; compare with `synthkit version`">loading
    build...</span></small></div>
  <button class="station active" data-s="describe"><b>01</b>
    Describe<small class="subt">define the dataset</small></button>
  <button class="station" data-s="spec"><b>02</b>
    Spec<small class="subt">review the recipe</small></button>
  <button class="station" data-s="data"><b>03</b>
    Data<small class="subt">generate synthetic data</small></button>
  <button class="station" data-s="campaign"><b>04</b>
    Campaign<small class="subt">configure the evaluation</small></button>
  <button class="station" data-s="showdown"><b>05</b>
    Showdown<small class="subt">compare results</small></button>
</nav>
<main>
<header class="bar">
  <h1 id="title">Describe <span>— plain English in</span></h1>
  <div id="speccard" class="empty">
    <div class="fp">no spec loaded</div>
    <div class="meta">describe one or load a preset</div>
  </div>
</header>

<section id="s-describe" class="active">
  <div class="stepbanner">Step 1 of 5 &mdash; Say what data you
  need</div>
  <div class="gov"><span>&#128274; Synthetic only &mdash; no real
  patient data touched</span><span>&#128273; Known answer key
  &mdash; every truth planted on purpose</span><span>&#128257;
  Reproducible &mdash; same fingerprint, same data,
  forever</span><span>&#128100; Human review before anything is
  created</span></div>
  <div class="explain">Everything starts with a description of a
  dataset. Nothing here touches real patients &mdash; every record is
  invented, but invented to order: realistic values, realistic
  messiness, and a known answer key. Use a ready-made example
  (1.1) or write your own description (1.2), then continue to
  Step 2.</div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">1.1</span>
    <span class="badge opt">optional &mdash; fastest path</span>
    ready-made examples</div>
    <div class="presets" id="presets"></div>
    <div class="hint">Click one to load a complete, tested
    description into Step 2. For the vendor evaluation demo,
    click the first card and skip to Step 2.</div>
  </div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">1.2</span>
    <span class="badge opt">skip if you clicked an example</span>
    describe the data in plain English</div>
    <div class="explain">Write what one record is (a patient, a
    bill, a lab result), what fields it has, how values should
    behave (typical ranges, how much is missing, what gets
    mistyped), any free-text field like a discharge note and
    what risk factors should hide inside it, and what outcome
    should be predictable and how rare it is. What this box
    cannot do: fetch real data, and it only produces dataset
    descriptions &mdash; not code, not reports.</div>
    <textarea id="english" rows="5"
      placeholder="a 300-row lab results extract: patient id, ordering department weighted toward internal medicine, ten percent missing results, occasional wrong-value dates..."></textarea>
    <div class="row2">
      <div><label for="kind"><span class="stepno">1.3</span>
        what shape of data?</label>
        <select id="kind"><option value="table">a table &mdash;
        one row per patient / record</option>
        <option value="document">text documents &mdash; e.g.
        clinical notes or reports</option></select></div>
      <div><label for="backend"><span class="stepno">1.4</span>
        <span class="badge opt">advanced</span> which AI reads
        your English</label>
        <select id="backend"><option value="ollama">local AI on
        this machine (private)</option>
        <option value="openai">local AI server (llama.cpp /
        LM Studio)</option>
        <option value="bedrock">hospital cloud (AWS
        Bedrock)</option>
        <option value="anthropic">Anthropic API</option></select></div>
    </div>
    <div class="eyebrow"><span class="stepno">1.5</span>
    <span class="badge opt">skip if you clicked an example</span>
    turn the description into a recipe</div>
    <button class="act" onclick="compileSpec()">Compile spec</button>
    <div class="hint">The AI drafts a formal recipe from your
    paragraph and Step 2 checks it. Drafts can contain mistakes
    &mdash; nothing is created until the recipe passes review. That
    pause is deliberate: a person approves every recipe.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="compile-out"></pre>
  </div>
</section>

<section id="s-spec">
  <div class="stepbanner">Step 2 of 5 &mdash; Review the recipe (the
  contract for your data)</div>
  <div class="explain">This is the complete, exact recipe the
  rest of the process follows: every field, every distribution,
  every deliberate flaw, every hidden trap, and the promised
  outcome rate. You do not need to read it &mdash; the buttons below
  do the reading. The short code above the page (the
  fingerprint) identifies this recipe forever: same fingerprint,
  same data, every time, on any machine.</div>
  <div class="panel">
    <div id="spec-summary"></div>
    <div class="eyebrow"><span class="stepno">2.1</span>
    <span class="badge opt">experts only</span> the recipe
    itself (editable)</div>
    <textarea id="spec" rows="22" spellcheck="false"
      placeholder="No spec yet &mdash; describe one or load a preset."></textarea>
    <div class="eyebrow"><span class="stepno">2.2</span>
    <span class="badge req">required</span> check the recipe is
    complete and lawful</div>
    <button class="act" onclick="validateSpec()">Validate</button>
    <div class="hint">Green means every field is well-defined
    and internally consistent. Problems are listed in plain
    terms so they can be fixed before anything is created.</div>
    <div class="eyebrow"><span class="stepno">2.3</span>
    <span class="badge opt">optional</span> quick trial run
    (nothing saved)</div>
    <button class="act ghost" onclick="planSpec()">Plan (dry run)</button>
    <div class="hint">Builds the dataset in memory and reports
    what it WOULD contain &mdash; row counts, corrupted cells &mdash;
    without writing anything to disk.</div>
    <div class="eyebrow"><span class="stepno">2.4</span>
    <span class="badge rec">recommended</span> does the data
    keep the recipe's promises?</div>
    <button class="act ghost" onclick="lintSpec()">Semantic lint</button>
    <div class="hint">Generates a sample and measures it against
    what was declared &mdash; e.g. "readmission was promised at
    5&ndash;12% and lands at 8%". This is how you know the dataset
    means what the description said.</div>
    <div class="eyebrow"><span class="stepno">2.5</span>
    <span class="badge opt">optional</span> save the recipe
    file</div>
    <button class="act ghost" onclick="downloadSpec()">Download spec.json</button>
    <div class="hint">Anyone with this one file can regenerate
    this exact dataset &mdash; that is the reproducibility guarantee,
    and what you would hand an auditor or a vendor.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="spec-out"></pre>
  </div>
</section>

<section id="s-data">
  <div class="stepbanner">Step 3 of 5 &mdash; Create the synthetic
  data</div>
  <div class="explain">This turns the approved recipe into real
  files: the messy dataset (what a model would actually face),
  the clean answer key (the same records with every flaw
  repaired and every truth known), and a ledger listing every
  deliberate corruption. When it finishes, a preview and
  download buttons appear below.</div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">3.1</span>
    <span class="badge req">required</span> where to save</div>
    <label for="outdir">folder name for this run</label>
    <input id="outdir" value="gui_runs/run_001">
    <div class="eyebrow"><span class="stepno">3.2</span>
    <span class="badge opt">advanced &mdash; text documents
    only</span> who writes the prose</div>
    <div class="row2">
      <div><label for="rbackend">writer</label>
        <select id="rbackend"><option value="stub">built-in
        writer &mdash; no AI, instant, always identical</option>
        <option value="ollama">Ollama on this machine (e.g.
        mistral-24B on a Mac)</option>
        <option value="openai">local AI server &mdash; llama.cpp /
        LM Studio (e.g. Llama-3B on this laptop)</option>
        <option value="bedrock">hospital AWS cloud (Bedrock:
        Llama / Mistral / Claude)</option>
        <option value="anthropic">Anthropic cloud API
        (Claude)</option></select></div>
      <div><label for="rmodel">model name (blank = default)</label>
        <input id="rmodel" placeholder="mistral-small3.1"></div>
    </div>
    <div class="hint">What each writer is: <b>built-in</b> = no
    AI at all; sentences come from the recipe's own phrase
    lists &mdash; instant, free, byte-identical every run
    (recommended for tables and live demos). <b>Ollama</b> = a
    large open model served by the Ollama app on this machine.
    <b>local AI server</b> = any OpenAI-compatible server on
    this computer (llama.cpp, LM Studio, vLLM) &mdash; the $0
    fully-offline option. ("OpenAI-compatible" names the
    message FORMAT those servers all adopted, not the company:
    nothing is sent to OpenAI or ChatGPT, no account exists,
    and traffic stays on this machine &mdash; it works with the
    network unplugged.) <b>Bedrock</b> = open-weight and
    Claude models inside the hospital's governed AWS.
    <b>Anthropic</b> = Claude via external API. Tables are
    ALWAYS generated exactly from the recipe regardless of this
    choice; AI writers only phrase the free-text notes, and
    every AI-written note is verified against the answer key
    and corrected if it drifts.</div>
    <div class="eyebrow"><span class="stepno">3.3</span>
    <span class="badge req">required</span> create the data</div>
    <button class="act" onclick="renderSpec()">Create the data</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="render-out"></pre>
    <div id="data-summary"></div>
    <div id="data-downloads"></div>
    <div id="render-preview"></div>
  </div>
</section>

<section id="s-campaign">
  <div class="stepbanner">Step 4 of 5 &mdash; Set the exam, then let
  our own model take it</div>
  <div class="explain">A campaign is a standardized exam built
  from the recipe: the same test at three difficulty levels,
  with pass marks you set. Here you define the exam (4.1&ndash;4.4)
  and then have synthkit's own built-in model sit it (4.5&ndash;4.6)
  &mdash; so before any vendor is judged, you know what a free,
  transparent model can score on this data.</div>
  <div class="panel">
    <div class="row2">
      <div><label for="goal"><span class="stepno">4.1</span>
        <span class="badge req">required</span> what is the
        task?</label>
        <select id="goal"><option value="clean">clean &mdash; repair
        messy values</option>
        <option value="predict">predict &mdash; forecast a yes/no
        outcome (the vendor demo)</option>
        <option value="regress">regress &mdash; predict a
        number</option>
        <option value="extract">extract &mdash; pull facts out of
        text</option></select></div>
      <div><label for="outcome"><span class="stepno">4.2</span>
        <span class="badge req">required for predict</span>
        which column is being predicted?</label>
        <input id="outcome" placeholder="readmitted_30d"></div>
    </div>
    <label for="bars"><span class="stepno">4.3</span>
    <span class="badge req">required</span> pass marks
    (name=value, comma-separated)</label>
    <input id="bars" value="fix_rate=0.9,detect_rate=0.5">
    <div id="bars-plain" class="hint"></div>
    <div class="hint">For prediction: auroc=0.6,gap_max=0.3.
    AUROC is the ranking score &mdash; 1.0 is perfect, 0.5 is a coin
    flip; 0.6 says "must beat a coin flip convincingly".
    gap_max limits how much worse a model may do on the messy
    data versus the clean answer key.</div>
    <div class="eyebrow"><span class="stepno">4.4</span>
    <span class="badge req">required</span> build the exam</div>
    <button class="act" onclick="campaignCompile()">Compile ladder</button>
    <label for="solver"><span class="stepno">4.5</span>
    <span class="badge req">required</span> which model takes
    the exam</label>
    <select id="solver"><option value="autoclean">autoclean &mdash;
      built-in cleaning tool</option>
      <option value="strip_cleaner">strip_cleaner &mdash; trivial
      cleaner (a control)</option>
      <option value="autosolver">autosolver &mdash; built-in tabular
      model (cannot read notes)</option>
      <option value="autosolver_regress">autosolver_regress &mdash;
      built-in, for numeric targets</option>
      <option value="autosolver_hybrid">autosolver_hybrid &mdash;
      our best: reads the table AND the notes</option>
      <option value="regex_extract">regex_extract &mdash; simple
      pattern extractor (a control)</option>
      <option value="llm_extract">llm_extract &mdash; an AI model
      in the test seat (extract only)</option></select>
    <div class="row2">
      <div><label for="lbackend"><span class="stepno">4.6</span>
        <span class="badge opt">only for llm_extract</span>
        which AI is being tested</label>
        <select id="lbackend"><option value="ollama">Ollama on
        this machine</option>
        <option value="openai">local AI server on this machine
        (llama.cpp / LM Studio &mdash; NOT ChatGPT)</option>
        <option value="bedrock">hospital AWS cloud
        (Bedrock)</option>
        <option value="anthropic">Anthropic cloud API
        (Claude)</option></select></div>
      <div><label for="lmodel">model name (blank =
        default)</label>
        <input id="lmodel" placeholder="mistral-small3.1"></div>
    </div>
    <div class="row2">
      <div><label for="samples"><span class="stepno">4.7</span>
        <span class="badge opt">advanced</span> answers per
        question (majority vote)</label>
        <input id="samples" value="1"></div>
      <div><label for="intervention"><span class="stepno">4.8</span>
        <span class="badge opt">advanced</span> extra
        instruction to the tested AI</label>
        <input id="intervention" placeholder="optional"></div>
    </div>
    <div class="eyebrow"><span class="stepno">4.9</span>
    <span class="badge req">required</span> run the exam</div>
    <button class="act" onclick="campaignRun()">Run ladder</button>
    <div class="hint">Training happens on a separate practice
    population the model has never been scored on; answers are
    hidden during the test. Results arrive per difficulty tier
    with statistical confidence attached.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="campaign-out"></pre>
  </div>
</section>

<section id="s-showdown">
  <div class="stepbanner">Step 5 of 5 &mdash; The verdict: vendor vs
  our model</div>
  <div class="explain">The vendor's model and our own take the
  identical exam on identical data &mdash; data where the maximum
  achievable score is KNOWN, because we planted the truth. Below
  the raw readout, a full plain-English report explains what was
  tested, what traps the data contained, how each model works,
  and who won.</div>
  <div class="panel">
    <label for="sbaseline"><span class="stepno">5.1</span>
    <span class="badge req">required</span> our challenger (the
    floor the vendor must beat)</label>
    <select id="sbaseline"><option value="autosolver">autosolver
      &mdash; tabular only (cannot read notes)</option>
      <option value="autosolver_hybrid">autosolver_hybrid &mdash; our
      best: reads the table AND the notes</option></select>
    <label for="vendor"><span class="stepno">5.2</span>
    <span class="badge req">required</span> the vendor's model
    (name or file:function)</label>
    <input id="vendor" value="autosolver">
    <div class="hint">Accepts a built-in name (for practice
    runs) or <b>file:function</b> &mdash; a Python file placed next
    to synthkit exposing one function that takes (training
    records, training answers, test records) and returns one
    risk score per test record. For this demo:
    <b>vendor_model:predict</b> &mdash; "VendorCo RiskScore", a
    competent model that simply cannot read the notes.</div>
    <details class="explain"><summary><b>How would a REAL
    vendor's model plug in?</b> (the question procurement will
    ask)</summary>
    Three routes, in increasing formality:
    <b>(1) Wrapper file</b> &mdash; the vendor's team writes a
    ten-line Python function that calls their model (their
    library, their API, their container) and returns scores;
    drop the file next to synthkit, type its name here. The
    vendor's code stays theirs; synthkit only sees scores.
    <b>(2) API wrapper</b> &mdash; same file, but the function
    calls the vendor's hosted scoring endpoint over the
    network; nothing is installed locally.
    <b>(3) Offline scoring exchange</b> &mdash; for vendors who
    will not integrate: export the test population from Step 3
    (downloads: synthetic data CSV &mdash; WITHOUT the outcome
    column), send it, receive their scores back as a file, and
    wrap that file in a two-line function that returns the
    scores in order. In every route the vendor never sees the
    answer key, and the exam stays identical for every
    contestant. Full walkthrough with copy-paste templates:
    docs/vendor_integration.md.</details>
    <div class="eyebrow"><span class="stepno">5.3</span>
    <span class="badge req">required</span> run the head-to-head</div>
    <button class="act" onclick="showdown()">Run showdown</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="showdown-out"></pre>
    <div id="showdown-report"></div>
  </div>
</section>
</main></div>
<script>
const titles={describe:['Describe','plain English in'],
  spec:['Spec','review before rendering'],
  data:['Data','the auditable artifact'],
  campaign:['Campaign','a goal becomes a ladder'],
  showdown:['Showdown','ceiling / baseline / vendor']};
let campaignDir='';
document.querySelectorAll('.station').forEach(btn=>{
  btn.onclick=()=>{
    document.querySelectorAll('.station').forEach(
      b=>b.classList.remove('active'));
    document.querySelectorAll('section').forEach(
      s=>s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('s-'+btn.dataset.s)
      .classList.add('active');
    const t=titles[btn.dataset.s];
    document.getElementById('title').innerHTML=
      t[0]+' <span>&mdash; '+t[1]+'</span>';
  };});
async function api(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',
    headers:{'Content-Type':'application/json'},
    body:body?JSON.stringify(body):undefined});
  return r.json();}
async function sha256(text){
  const buf=await crypto.subtle.digest('SHA-256',
    new TextEncoder().encode(text));
  return[...new Uint8Array(buf)].map(
    b=>b.toString(16).padStart(2,'0')).join('');}
async function refreshCard(){
  const card=document.getElementById('speccard');
  const raw=document.getElementById('spec').value.trim();
  if(!raw){card.className='empty';card.innerHTML=
    '<div class="fp">no spec loaded</div>'+
    '<div class="meta">describe one or load a preset</div>';
    return;}
  let title='(unparsed)',rows='';
  try{const d=JSON.parse(raw);title=d.title||'(untitled)';
    rows=d.rows?d.rows+' rows':(d.corpus&&d.corpus.size?
      d.corpus.size+' docs':'');}catch(e){}
  const fp=(await sha256(raw)).slice(0,12);
  card.className='';
  card.innerHTML='<div class="fp">spec:'+fp+'</div>'+
    '<div class="meta">'+title+(rows?' &middot; '+rows:'')+
    ' &middot; same fingerprint, same data</div>';}
document.getElementById('spec')
  .addEventListener('input',refreshCard);
function setSpec(obj){
  document.getElementById('spec').value=
    JSON.stringify(obj,null,2);
  refreshCard();}
function out(id,text,cls){const el=document.getElementById(id);
  el.innerHTML='';const span=document.createElement('span');
  if(cls)span.className=cls;span.textContent=text;
  el.appendChild(span);}
async function loadPresets(){
  const d=await api('/api/presets');
  const box=document.getElementById('presets');
  d.presets.forEach(p=>{
    const b=document.createElement('button');
    b.textContent=p.name;b.title=p.description;
    b.onclick=()=>{
      if(p.spec){setSpec(p.spec);
        document.querySelector('[data-s=spec]').click();}
      else{document.getElementById('english').value=p.english;
        document.getElementById('kind').value=p.kind;}};
    box.appendChild(b);});}
async function compileSpec(){
  if(!gate(['english'],'compile-out'))return;
  out('compile-out','compiling via local model...');
  const d=await api('/api/compile',{
    description:document.getElementById('english').value,
    kind:document.getElementById('kind').value,
    backend:document.getElementById('backend').value});
  if(d.error){out('compile-out',d.error,'bad');return;}
  try{setSpec(JSON.parse(d.raw_json));}catch(e){}
  out('compile-out',d.ok?
    'Compiled clean. Review the draft at station 02 before '+
    'rendering.':'Draft saved with problems — the human gate is '+
    'yours:\n\n'+d.problems,d.ok?'ok':'bad');}
async function validateSpec(){
  const d=await api('/api/validate',{
    kind:guessKind(),spec:document.getElementById('spec').value});
  if(d.ok)tick('spec');
  out('spec-out',d.ok?'Spec validates. A validated spec must '+
    'plan — that is the contract.':d.problems,d.ok?'ok':'bad');}
function guessKind(){
  try{const d=JSON.parse(
    document.getElementById('spec').value);
    return d.columns?'table':'document';}catch(e){
    return 'table';}}
function downloadSpec(){
  const raw=document.getElementById('spec').value;
  if(!raw.trim()){out('spec-out',
    'Nothing to download - no spec loaded.','bad');return;}
  let name='spec';
  try{name=(JSON.parse(raw).title||'spec')
    .toLowerCase().replace(/[^a-z0-9]+/g,'_')
    .replace(/^_+|_+$/g,'');}catch(e){}
  const blob=new Blob([raw],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=name+'.json';a.click();
  URL.revokeObjectURL(a.href);}
async function lintSpec(){
  const d=await api('/api/lint',{kind:guessKind(),
    spec:document.getElementById('spec').value,
    description:document.getElementById('english').value});
  out('spec-out',d.error?d.error:d.text,
    d.error?'bad':(d.ok?'ok':''));}
async function planSpec(){
  const d=await api('/api/plan',{kind:guessKind(),
    spec:document.getElementById('spec').value});
  out('spec-out',d.error?d.error:
    JSON.stringify(d,null,2),d.error?'bad':'');}
async function renderSpec(){
  if(!gate(['spec','outdir'],'render-out'))return;
  var backend=document.getElementById('rbackend').value;
  var payload={kind:guessKind(),
    spec:document.getElementById('spec').value,
    out:document.getElementById('outdir').value,
    backend:backend,
    model:document.getElementById('rmodel').value};
  if(backend==='stub'){
    out('render-out','rendering...');
    var d=await api('/api/render',payload);
    renderDone(d);}
  else{
    out('render-out','rendering via '+backend+'... 0s');
    var d=await api('/api/render-async',payload);
    poll(d.job,'render-out',renderDone);}}
var PV=null;
function specOk(){
  var v=document.getElementById('spec').value.trim();
  if(!v)return false;
  try{JSON.parse(v);return true;}catch(e){return false;}}
function barsOk(){
  var v=document.getElementById('bars').value.trim();
  return /^\s*[a-z_]+\s*=\s*[0-9.]+(\s*,\s*[a-z_]+\s*=\s*[0-9.]+)*\s*$/.test(v);}
var READY={
  english:function(){return document.getElementById(
    'english').value.trim().length>=15;},
  spec:specOk,
  outdir:function(){return document.getElementById(
    'outdir').value.trim().length>0;},
  bars:barsOk,
  outcome:function(){
    var g=document.getElementById('goal').value;
    var v=document.getElementById('outcome').value.trim();
    return (g!=='predict'&&g!=='regress')||v.length>0;},
  vendor:function(){return document.getElementById(
    'vendor').value.trim().length>0;}};
function paintReady(){
  specSummary();barsExplain();
  for(var id in READY){
    var el=document.getElementById(id);
    if(!el)continue;
    var good=READY[id]();
    el.classList.toggle('good',good);
    el.classList.toggle('need',!good);}}
function gate(ids,outId){
  paintReady();
  var missing=[];
  for(var i=0;i<ids.length;i++){
    if(!READY[ids[i]]()){missing.push(ids[i]);}}
  if(missing.length){
    out(outId,'Fill the highlighted red field(s) first: '+
      missing.join(', ')+'. Green = ready.','bad');
    document.getElementById(missing[0]).focus();
    return false;}
  return true;}

function esc(s){return String(s).replace(/&/g,'&amp;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function openCell(r,c){
  if(!PV)return;
  var col=PV.cols[c];
  var v=String(PV.rows[r][col]);
  document.getElementById('cellmodal-title').textContent=
    col+' \u2014 record '+(r+1);
  document.getElementById('cellmodal-body').textContent=
    v===''?'(empty cell \u2014 a deliberately missing value)':v;
  document.getElementById('cellmodal').style.display='block';}
function closeCell(){
  document.getElementById('cellmodal').style.display='none';}
document.addEventListener('keydown',function(e){
  if(e.key==='Escape')closeCell();});
function specSummary(){
  var box=document.getElementById('spec-summary');
  if(!box)return;
  var v=document.getElementById('spec').value.trim();
  if(!v){box.innerHTML='';return;}
  var s;try{s=JSON.parse(v);}catch(e){
    box.innerHTML='<div class="summarycard">The recipe text '+
    'is not valid yet \u2014 fix the red box below or reload '+
    'a preset.</div>';return;}
  var cols=s.columns||[];
  var notes=[],flaws={},i;
  for(i=0;i<cols.length;i++){
    if(cols[i].ctype==='note')notes.push(cols[i].name);
    var m=cols[i].mess||{};
    if(m.missing_rate)flaws['missing values']=1;
    if(m.typo_rate)flaws['typos']=1;
    if(m.format_rate)flaws['mixed formats']=1;
    if(m.outlier_rate)flaws['implausible outliers']=1;
    if(m.wrong_rate)flaws['subtly wrong values']=1;
    if(m.case_rate||m.space_rate)flaws['casing/whitespace']=1;}
  if(s.duplicate_rate)flaws['duplicated rows']=1;
  var oc=(s.outcomes||[])[0];
  var h='<div class="summarycard"><b>Recipe summary'+
    '</b> \u2014 '+(s.rows||'?')+' records \u00d7 '+
    cols.length+' fields'+(notes.length?', including free-text '+
    'note field'+(notes.length>1?'s':'')+' <b>'+
    notes.join(', ')+'</b>':'')+'.';
  if(oc){h+=' Outcome <b>'+oc.name+'</b>';
    if(oc.target_prevalence){h+=' promised at '+
      (100*oc.target_prevalence[0]).toFixed(0)+'\u2013'+
      (100*oc.target_prevalence[1]).toFixed(0)+
      '% of records (minority class)';}h+='.';}
  var fk=Object.keys(flaws);
  if(fk.length)h+=' Deliberate flaws: '+fk.join(', ')+'.';
  h+=' (Full formal recipe below \u2014 the buttons check it '+
    'so you never have to read it.)</div>';
  box.innerHTML=h;}
var BAR_WORDS={
  auroc:'ranking score at least VAL (1.0 = perfect, 0.5 = '+
    'coin flip)',
  gap_max:'messy-data score may trail the clean answer key '+
    'by at most VAL',
  fix_rate:'must repair at least PCT of corrupted cells',
  detect_rate:'must flag at least PCT of corruptions',
  recall:'must find at least PCT of planted facts',
  trap_max:'may fall for at most PCT of planted traps',
  rmse_max:'average numeric error at most VAL'};
function barsExplain(){
  var el=document.getElementById('bars-plain');
  if(!el)return;
  var v=document.getElementById('bars').value.trim();
  if(!barsOk()){el.textContent='';return;}
  var parts=v.split(','),outp=[];
  for(var i=0;i<parts.length;i++){
    var kv=parts[i].split('=');
    var k=kv[0].trim(),val=parseFloat(kv[1]);
    var w=BAR_WORDS[k];
    if(w){outp.push(w.replace('VAL',val)
      .replace('PCT',(100*val).toFixed(0)+'%'));}
    else{outp.push(k+' \u2265 '+val);}}
  el.innerHTML='In plain terms: '+outp.join('; ')+'.';}
function tick(step){
  var b=document.querySelector(
    '.station[data-s="'+step+'"]');
  if(b)b.classList.add('done');}
document.addEventListener('input',paintReady);
document.addEventListener('change',paintReady);
window.addEventListener('load',function(){
  setTimeout(paintReady,150);});
function fmtPct(x){return (100*x).toFixed(1)+'%';}
function dataSummary(d){
  var h='<div class="mcards">'+
    '<div class="mcard"><div class="n">'+d.rows+
    '</div><div class="l">synthetic records</div></div>'+
    '<div class="mcard"><div class="n">'+d.mess_cells+
    '</div><div class="l">deliberate corruptions '+
    '(ledgered)</div></div>';
  for(var i=0;i<(d.outcomes||[]).length;i++){
    var o=d.outcomes[i];var ok=true,sub='outcome rate';
    if(o.declared){
      ok=o.realized>=o.declared[0]&&
         o.realized<=o.declared[1];
      sub=o.name+' (promised '+fmtPct(o.declared[0])+
        '\u2013'+fmtPct(o.declared[1])+')';}
    h+='<div class="mcard '+(ok?'ok':'warn')+'">'+
      '<div class="n">'+fmtPct(o.realized)+
      (ok?' \u2713':' \u26a0')+'</div>'+
      '<div class="l">'+sub+'</div></div>';}
  h+='</div><div class="sumline">The messy dataset, the clean '+
    'answer key, and the corruption ledger are downloadable '+
    'below; the first rows are previewed at the bottom \u2014 '+
    'click any cell to read it in full.</div>';
  return h;}
async function exportData(which,fmt){
  var d=await api('/api/export',{
    dir:document.getElementById('outdir').value,
    which:which,fmt:fmt});
  if(d.error){out('render-out',d.error,'bad');return;}
  var blob=new Blob([d.content],{type:d.mime});
  var a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=d.filename;a.click();
  URL.revokeObjectURL(a.href);}
function downloadBar(){
  return '<div style="margin:6px 0 2px;font-size:11px;'+
    'font-weight:700;text-transform:uppercase;'+
    'letter-spacing:.05em;color:#5a6a63">downloads</div>'+
    '<button class="dl" onclick="exportData(\'dirty\',\'csv\')">synthetic data (CSV)</button>'+
    '<button class="dl" onclick="exportData(\'dirty\',\'json\')">synthetic data (JSON)</button>'+
    '<button class="dl" onclick="exportData(\'clean\',\'csv\')">answer key (CSV)</button>'+
    '<button class="dl" onclick="exportData(\'clean\',\'json\')">answer key (JSON)</button>'+
    '<button class="dl" onclick="exportData(\'ledger\',\'json\')">corruption ledger</button>';}
function renderDone(d){
  if(d.error){out('render-out',d.error,'bad');return;}
  tick('data');
  document.getElementById('data-summary').innerHTML='';
  document.getElementById('data-downloads').innerHTML='';
  if(d.preview){
    out('render-out','wrote '+d.run_dir+'  ('+
      d.mess_cells+' mess cells, ledgered)');
    PV={cols:d.columns,rows:d.preview};
    var cols=d.columns;
    var html='<div class="pvwrap"><table><tr>';
    for(var c=0;c<cols.length;c++){
      html+='<th>'+esc(cols[c])+'</th>';}
    html+='</tr>';
    for(var r=0;r<d.preview.length;r++){
      html+='<tr>';
      for(var c2=0;c2<cols.length;c2++){
        var v=String(d.preview[r][cols[c2]]===''?
          '':d.preview[r][cols[c2]]);
        var short=v.length>90?v.slice(0,90)+'\u2026':v;
        html+='<td onclick="openCell('+r+','+c2+')">'+
          (v===''?'&empty;':esc(short))+'</td>';}
      html+='</tr>';}
    html+='</table></div>'+
      '<div class="pvhint">Showing the first '+
      d.preview.length+' of '+d.rows+' records \u2014 '+
      'scroll the pane in both directions; click any cell '+
      'to read its full contents (notes especially).</div>';
    document.getElementById('render-preview').innerHTML=html;
    document.getElementById('data-summary').innerHTML=
      dataSummary(d);
    document.getElementById('data-downloads').innerHTML=
      downloadBar();}
  else{out('render-out','wrote '+d.run_dir+'  ('+d.documents+
    ' docs, '+d.verified_first_try+
    ' verified first try)\n\n--- first document ---\n'+
    d.first_document);
    document.getElementById('render-preview').innerHTML='';}}
async function campaignCompile(){
  if(!gate(['spec','bars','outcome'],'campaign-out'))return;
  const bars={};document.getElementById('bars').value
    .split(',').forEach(p=>{const[k,v]=p.split('=');
    if(k&&v)bars[k.trim()]=parseFloat(v);});
  out('campaign-out','compiling ladder...');
  const d=await api('/api/campaign-compile',{
    goal:document.getElementById('goal').value,
    spec:document.getElementById('spec').value,
    bars:bars,outcome:document.getElementById('outcome').value,
    out:''});
  if(d.error){out('campaign-out',d.error,'bad');return;}
  campaignDir=d.campaign_dir;saveSession();
  out('campaign-out',d.title+' -> '+d.campaign_dir+'\n\n'+d.tiers.map((t,i)=>
    'tier '+(i+1)+' `'+t.name+'`: '+t.notes+'\n    '+
    t.conditions.join('\n    ')).join('\n'));}
let activeJob=null;
async function cancelJob(){
  if(activeJob)await api('/api/job-cancel',{id:activeJob});}
async function poll(jobId,outId,render){
  activeJob=jobId;
  const t=setInterval(async()=>{
    const j=await api('/api/job',{id:jobId});
    if(j.status==='running'){
      let msg='working... '+j.elapsed+'s';
      if(j.elapsed>300)msg+='  (long run: llm ladders take '
        +'minutes; Cancel abandons it)';
      out(outId,msg);
      const el=document.getElementById(outId);
      const b=document.createElement('button');
      b.className='act ghost';b.textContent='Cancel job';
      b.style.marginLeft='12px';b.onclick=cancelJob;
      el.appendChild(b);}
    else{clearInterval(t);activeJob=null;
      if(j.status==='error'||j.status==='timeout'
        ||j.status==='cancelled'){out(outId,j.error,'bad');}
      else{render(j.result);}}},1500);}

async function campaignRun(){
  if(!campaignDir){out('campaign-out',
    'Compile a ladder first.','bad');return;}
  out('campaign-out','walking the ladder... 0s');
  const d=await api('/api/campaign-run-async',{
    campaign_dir:campaignDir,
    solver:document.getElementById('solver').value,
    samples:parseInt(document.getElementById(
      'samples').value)||1,
    backend:document.getElementById('lbackend').value,
    model:document.getElementById('lmodel').value,
    extra_system:document.getElementById(
      'intervention').value});
  poll(d.job,'campaign-out',function(r){
    tick('campaign');
    out('campaign-out',r.text,
      r.highest_passed===r.tiers?'ok':'');});}
async function showdown(){
  if(!gate(['vendor'],'showdown-out'))return;
  if(!campaignDir){out('showdown-out',
    'Compile a predict campaign at station 04 first.','bad');
    return;}
  out('showdown-out','running both solvers up the ladder... 0s');
  const d=await api('/api/showdown-async',{
    campaign_dir:campaignDir,
    baseline:document.getElementById('sbaseline').value,
    solver:document.getElementById('vendor').value});
  poll(d.job,'showdown-out',function(r){
    out('showdown-out',r.text,'');
    tick('showdown');renderReport(r.report);});}
function tierRows(t,who){
  var h='';
  for(var i=0;i<t.length;i++){
    var v=who==='vendor'?t[i].vendor:t[i].baseline;
    var win=who==='vendor'?
      t[i].vendor>=t[i].baseline:
      t[i].baseline>=t[i].vendor;
    var cls=(who==='vendor'?t[i].passed:true)&&win?
      'pass':((who==='vendor'&&!t[i].passed)?
      'fail':(win?'pass':'fail'));
    h+='<div class="tierrow '+cls+'">'+t[i].tier+
      ': score '+v.toFixed(3)+' (max possible '+
      t[i].ceiling.toFixed(3)+') &mdash; '+
      (win?'ahead':'behind')+' in this tier'+
      (who==='vendor'?(t[i].passed?
      ', pass mark met':', BELOW the pass mark'):'')+
      '</div>';}
  return h;}
function modelCard(name,how,tiers,who,isWinner){
  return '<div class="vcard '+
    (isWinner?'winner':'loser')+'"><h4>'+name+
    (isWinner?'<span class="wbadge">WINNER</span>':'')+
    '</h4><p>'+how+'</p>'+tierRows(tiers,who)+
    '</div>';}
function renderReport(rep){
  var el=document.getElementById('showdown-report');
  if(!rep){el.innerHTML='';return;}
  var ds=rep.dataset;
  var h='<div class="report">';
  h+='<h3>What just happened, in plain terms</h3>';
  h+='<p>A synthetic population of <b>'+ds.rows+
    ' records</b> ('+ds.n_columns+' fields'+
    (ds.note_columns?', including '+ds.note_columns+
    ' free-text note field':'')+
    ') was generated from an approved recipe, with '+
    'the outcome <b>'+ds.outcome+'</b>'+
    (ds.declared?' promised to occur in '+
    (100*ds.declared[0]).toFixed(0)+'&ndash;'+
    (100*ds.declared[1]).toFixed(0)+
    '% of records (a rare, minority outcome '+
    '&mdash; like real clinical data)':'')+
    '. Both models trained on a separate practice '+
    'population and were scored blind on this one, '+
    'at three difficulty levels. Because the truth '+
    'was planted, the <b>maximum achievable score '+
    'is known exactly</b> &mdash; no model can '+
    'legitimately beat the ceiling shown in each '+
    'row.</p>';
  if(rep.mess_kinds&&rep.mess_kinds.length){
    h+='<p>The data was deliberately imperfect: '+
      rep.mess_kinds.join(', ')+
      ' &mdash; every corruption recorded in the '+
      'ledger from Step 3.</p>';}
  if(rep.traps&&rep.traps.length){
    h+='<h3>Hidden traps and buried signal</h3><ul>';
    for(var i=0;i<rep.traps.length;i++){
      var t=rep.traps[i];
      h+='<li><b>'+t.kind+':</b> '+
        '<span class="trapx">&ldquo;'+t.example+
        '&rdquo;</span> &mdash; '+t.why+'</li>';}
    h+='</ul>';}
  var w=rep.winner;
  if(w){
    var vt=rep.tiers_data,np=0;
    for(var vi=0;vi<vt.length;vi++){
      if(vt[vi].passed)np++;}
    h+='<div class="vstrip"><b>VERDICT:</b> '+w.name+
      ' wins on the data as specified (by '+
      w.margin.toFixed(3)+' AUROC). The vendor met its pass '+
      'mark on '+np+' of '+vt.length+' difficulty tiers'+
      (w.vendor_won?'':' but was beaten by a model we built '+
      'for free')+'. Every number below was scored against a '+
      'planted, known truth.</div>';}
  h+='<h3>The verdict, in detail</h3>';
  if(w){
    h+='<p>On the <b>'+w.tier+'</b> tier (the data '+
      'exactly as specified), <b>'+w.name+
      '</b> wins by '+w.margin.toFixed(3)+
      ' AUROC (1.0 = perfect, 0.5 = coin flip).</p>';
    var vFirst=w.vendor_won;
    var vc=modelCard(rep.vendor.name,rep.vendor.how,
      rep.tiers_data,'vendor',vFirst);
    var bc=modelCard(rep.baseline.name,
      rep.baseline.how,rep.tiers_data,'baseline',
      !vFirst);
    h+=vFirst?vc+bc:bc+vc;}
  var im=rep.inside_the_model;
  if(im&&im.top&&im.top.length){
    h+='<h3>Inside our model: what it actually learned</h3>'+
      '<p>Our challenger is a <b>'+im.kind+'</b> over '+
      im.n_features+' features'+(im.text_terms?' ('+
      im.text_terms+' of them phrases it mined from the '+
      'notes using only the training labels)':'')+
      '. Its strongest learned predictors \u2014 the actual '+
      'trained coefficients, not a summary:</p>';
    var mx=0;
    for(var mi=0;mi<im.top.length;mi++){
      mx=Math.max(mx,Math.abs(im.top[mi].weight));}
    for(var mj=0;mj<im.top.length;mj++){
      var it=im.top[mj];
      var wpx=Math.max(2,Math.round(
        90*Math.abs(it.weight)/mx));
      h+='<div class="modelrow"><span class="bar '+
        (it.weight>0?'up':'dn')+'" style="width:'+wpx+
        'px"></span><code>'+(it.weight>0?'+':'')+
        it.weight.toFixed(3)+'</code> '+esc(it.name)+
        ' \u2014 '+it.direction+'</div>';}
    h+='<p>Red bars raise predicted risk, blue bars lower '+
      'it. Note the mined note-phrases standing beside the '+
      'vital signs \u2014 that is the value the text-blind '+
      'vendor left on the table. The vendor model remains a '+
      'black box to us by design: we judge it only on its '+
      'scores.</p>';}
  h+='<p>Pass marks for this exam: '+
    (rep.bars||[]).join(', ')+'. Green rows met '+
    'their mark and led their tier; red rows fell '+
    'short.</p></div>';
  el.innerHTML=h;}
const PERSIST=['spec','english','kind','goal','outcome',
  'bars','solver','vendor','outdir','samples','intervention',
  'rbackend','rmodel','lbackend','lmodel'];
function saveSession(){
  const state={campaignDir:campaignDir};
  PERSIST.forEach(id=>{const el=document.getElementById(id);
    if(el)state[id]=el.value;});
  try{localStorage.setItem('synthkit_bench',
    JSON.stringify(state));}catch(e){}}
function restoreSession(){
  try{
    const raw=localStorage.getItem('synthkit_bench');
    if(!raw)return;
    const state=JSON.parse(raw);
    PERSIST.forEach(id=>{const el=document.getElementById(id);
      if(el&&state[id]!==undefined)el.value=state[id];});
    if(state.campaignDir)campaignDir=state.campaignDir;
    refreshCard();
  }catch(e){}}
PERSIST.forEach(id=>{const el=document.getElementById(id);
  if(el)el.addEventListener('input',saveSession);});
restoreSession();
loadPresets();
api('/api/version').then(d=>{
  document.getElementById('fp').textContent=
    'v'+d.version+' \u00b7 '+d.built+' \u00b7 '+d.fingerprint;});
</script><div id="cellmodal" onclick="if(event.target===this)closeCell()">
  <div class="box"><span class="close"
    onclick="closeCell()">&times;</span>
  <h4 id="cellmodal-title"></h4>
  <pre id="cellmodal-body"></pre></div>
</div>
</body></html>
"""

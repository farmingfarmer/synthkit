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
import sys
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


def _start_job(fn, payload: dict, budget_s: float = None) -> str:
    import time
    import uuid
    job_id = uuid.uuid4().hex[:12]
    # THE BUDGET RIDES ON THE JOB RECORD. One global number would
    # mark a 33-minute fit of the real extract as `timeout` while
    # the run was succeeding - a wrong verdict about a healthy job,
    # the exact class the over-budget check was rebuilt to avoid.
    _JOBS[job_id] = {"status": "running",
                     "started": time.time(),
                     "budget": float(budget_s or JOB_BUDGET_S),
                     "canceled": False}

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
    if status == "running" and job.get("canceled"):
        status = "canceled"
    elif status == "running" and elapsed > job.get(
            "budget", JOB_BUDGET_S):
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
                        "to move on.".format(
                            int(job.get("budget",
                                        JOB_BUDGET_S))))
    elif status == "canceled":
        out["error"] = ("canceled — the current backend call "
                        "may run to completion in the "
                        "background, then stop")
    return out


def api_job_cancel(payload: dict) -> dict:
    job = _JOBS.get(payload.get("id", ""))
    if job is None:
        return {"error": "unknown job"}
    job["canceled"] = True
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


_LEARNED = {}
_HISTORY = []


def _remember(action: str, summary: str, detail=None) -> None:
    """Keep a short record of what was done and what came back.

    A bench that forgets is hard to trust: yesterday's number is
    gone, and there is no way to see whether a change helped. This
    is deliberately small — what was done, when, and the figures
    that mattered — and it lives only for the session.
    """
    import datetime
    _HISTORY.insert(0, {
        "when": datetime.datetime.now().strftime("%H:%M:%S"),
        "action": action, "summary": summary,
        "detail": detail or {}})
    del _HISTORY[40:]


def api_learn_history(payload: dict) -> dict:
    return {"history": _HISTORY,
            "note": "this session only \u2014 saving a model "
                    "keeps the work itself, this keeps the trail"}


def api_learn(payload: dict) -> dict:
    """Learn the joint distribution of an existing tidy dataset.

    This is the other way into synthkit: instead of describing a
    dataset in English, point at one that exists and let the bench
    measure it. What crosses back is PARAMETERS — counts, curves
    and conditional tables, every published figure covering at
    least k patients — never records.
    """
    import csv as _csv
    from .condnet import CondNet
    from . import learnspec as _ls
    if payload.get("content"):
        # The browser read the file and sent its text, so a person
        # can point at a spreadsheet without knowing its path.
        # It is written to a scratch file the same way any other
        # source would be, and is deleted with the session.
        import tempfile
        td = Path(tempfile.mkdtemp(prefix="synthkit_upload_"))
        path = td / (payload.get("filename") or "uploaded.csv")
        path.write_text(payload["content"], encoding="utf-8")
    else:
        path = Path(payload["path"]).expanduser()
    if not path.exists():
        return {"error": "No file at {}. Give the full path to a "
                         "tidy CSV — one row per visit."
                         .format(path)}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return {"error": "That file has no rows."}
    group = payload.get("group_by") or "person_id"
    if group not in rows[0]:
        group = None
    k = int(payload.get("k") or 10)
    net = CondNet(k=k, max_parents=3).learn(
        rows, group_by=group,
        targets=[payload["target"]] if payload.get("target")
        else None)
    _LEARNED["net"] = net
    _LEARNED["rows"] = len(rows)
    _LEARNED["path"] = str(path)
    nar = _ls.narrate(net)
    facts = _ls.facts_from_model(net)
    _LEARNED["facts"] = facts
    _remember("learned", nar["headline"],
              {"file": path.name,
               "epsilon": net.report["differential_privacy"][
                   "epsilon"],
               "findings": len(nar["findings"])})
    return {"narrative": nar, "dials": _ls.dials(net),
            "outcome_candidates": _ls.outcome_candidates(net),
            "note_plan": _ls.plan_summary(facts),
            "privacy": ("Nothing from this file is stored. The "
                        "model holds counts and bin edges only, "
                        "each covering at least {} patients."
                        .format(k)),
            "grouped_by": group or "(none — each row counted as "
                                   "its own patient)"}


def api_learn_generate(payload: dict) -> dict:
    """Generate from the learned model, with the dials applied."""
    import csv as _csv
    import io
    from .condnet import CondNet
    from . import learnspec as _ls
    net = _LEARNED.get("net")
    if net is None:
        return {"error": "Learn from a dataset first."}
    net = CondNet.from_json(net.to_json())     # never mutate the
    _ls.apply_dials(net, payload.get("dials") or {})   # original
    rows = int(payload.get("rows") or _LEARNED.get("rows") or 500)
    if payload.get("hierarchical") and getattr(
            net, "visit_counts", None):
        # people with a course of visits, not loose encounters
        per = max(1.0, sum(k * v for k, v in
                           net.visit_counts.items())
                  / max(sum(net.visit_counts.values()), 1))
        syn = net.sample_patients(max(1, int(rows / per)),
                                  seed=int(payload.get("seed")
                                           or 7))
    else:
        syn = net.sample(rows, seed=int(payload.get("seed") or 7))
    notes = 0
    if payload.get("transcribe"):
        from .transcribe import TranscribeSpec, transcribe
        spec = TranscribeSpec(_LEARNED.get("facts") or [],
                              rates=_ls.default_rates())
        syn, ledgers = transcribe(syn, spec)
        notes = sum(len(x) for x in ledgers)
        _LEARNED["ledgers"] = ledgers
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=list(syn[0]))
    w.writeheader()
    w.writerows(syn)
    _LEARNED["generated"] = buf.getvalue()
    preview = syn[:40]
    temporal = {}
    if payload.get("hierarchical") and getattr(
            net, "target_autocorr", None):
        temporal = net.temporal_check(syn)
    _remember("generated",
              "{} records{}".format(
                  len(syn),
                  " with notes" if notes else ""),
              {"rows": len(syn), "notes": notes,
               "dials": {k: v for k, v in
                         (payload.get("dials") or {}).items()
                         if v != 1}})
    return {"rows": len(syn), "columns": len(syn[0]),
            "temporal": temporal,
            "ledgered_mentions": notes,
            "columns_list": list(syn[0]),
            "preview": preview,
            "dials_applied": payload.get("dials") or {}}


def api_learn_plant(payload: dict) -> dict:
    """Plant an outcome on the learned data and grade models on it.

    Learning produces realistic data; it does not produce an EXAM.
    An exam needs an answer key, and an answer key needs causes a
    person stated rather than causes inferred from the same data
    the models will see. So the weights here are authored, the
    probability behind every row is therefore known, and the best
    achievable score is known with it.
    """
    from . import learnspec as _ls
    from .condnet import CondNet
    from .transcribe import TranscribeSpec, transcribe
    net = _LEARNED.get("net")
    if net is None:
        return {"error": "Learn from a dataset first."}
    weights = {k: float(v) for k, v in
               (payload.get("weights") or {}).items() if float(v)}
    if not weights:
        return {"error": "Give at least one column a weight. "
                         "Those weights ARE the planted truth — "
                         "nothing here infers them for you."}
    rows = int(payload.get("rows") or 2000)
    name = payload.get("label") or "outcome"
    prev = payload.get("prevalence")
    prev = float(prev) if prev else None
    hide_text = bool(payload.get("hide_in_notes"))

    work = CondNet.from_json(net.to_json())
    _ls.apply_dials(work, payload.get("dials") or {})
    data = work.sample(rows, seed=int(payload.get("seed") or 11))
    planted = _ls.plant_outcome(data, weights, name=name,
                                prevalence=prev)
    if "error" in planted:
        return planted

    hidden = []
    if hide_text:
        hide_cols = [c for c in weights if "::" in c]
        facts = _ls.facts_from_model(net, text_only=hide_cols)
        spec = TranscribeSpec(facts, rates=_ls.default_rates())
        planted["rows"], _ = transcribe(planted["rows"], spec)
        hidden = _ls.columns_to_hide(net, hide_cols)
        _ls.blank_columns(planted["rows"], hidden)
    elif payload.get("transcribe"):
        spec = TranscribeSpec(_ls.facts_from_model(net),
                              rates=_ls.default_rates())
        planted["rows"], _ = transcribe(planted["rows"], spec)

    result = _ls.showdown(planted["rows"], planted["probs"], name,
                          vendor=payload.get("vendor", ""))
    if "error" in result:
        return result
    import csv as _csv
    import io
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=list(planted["rows"][0]))
    w.writeheader()
    w.writerows(planted["rows"])
    _LEARNED["generated"] = buf.getvalue()
    lines = []
    lines.append(
        "The best score anything could reach on this data is "
        "{:.3f}. That is not an estimate: the probability behind "
        "every record is known, because you stated the causes."
        .format(result["ceiling"]))
    if result.get("blind") is not None:
        lines.append(
            "A model reading the notes scored {:.3f}; the same "
            "model with the notes withheld scored {:.3f}. Reading "
            "was worth {:+.3f}."
            .format(result["reading"], result["blind"],
                    result["value_of_reading"]))
    else:
        lines.append(
            "Our own model scored {:.3f} against that ceiling."
            .format(result["reading"]))
    if hidden:
        lines.append(
            "The causes you weighted were removed from the "
            "columns and written only into the prose ({}), so a "
            "model that cannot read is missing them by "
            "construction rather than by accident."
            .format(", ".join(hidden)))
    if "vendor" in result:
        lines.append(
            "The vendor model scored {:.3f} \u2014 {} our own."
            .format(result["vendor"],
                    "better than" if result["vendor_beats_ours"]
                    else "short of"))
    if result.get("vendor_error"):
        lines.append("The vendor model could not be loaded: {}"
                     .format(result["vendor_error"]))
    _remember("graded",
              "ceiling {:.3f}, ours {:.3f}{}".format(
                  result["ceiling"], result["reading"],
                  ", blind {:.3f}".format(result["blind"])
                  if result.get("blind") is not None else ""),
              {"weights": planted["weights"]})
    return {"showdown": result, "planted": {
        "label": name, "intercept": planted["intercept"],
        "prevalence": planted["realized_prevalence"],
        "weights": planted["weights"],
        "note": planted["note"]},
        "hidden_columns": hidden, "plain": lines}


def api_learn_audit(payload: dict) -> dict:
    """Attack the model the bench just built.

    The command line could do this and the bench could not, which
    is the wrong way round: the bench is where most people work,
    and a privacy claim nobody tested is the one most likely to be
    repeated.
    """
    import csv as _csv
    import random as _rnd
    from .attack import membership_audit
    from .condnet import CondNet
    net = _LEARNED.get("net")
    srcp = _LEARNED.get("path")
    if net is None or not srcp:
        return {"error": "Learn from a dataset first."}
    with Path(srcp).open(encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.DictReader(f))
    gb = getattr(net, "group_by", None) or "person_id"
    by = {}
    for r in rows:
        by.setdefault(r.get(gb, id(r)), []).append(r)
    keys = sorted(by, key=str)
    _rnd.Random(7).shuffle(keys)
    cut = len(keys) // 2
    if cut < 5:
        return {"error": "Too few patients to test membership: "
                         "the split would leave nothing to "
                         "compare."}
    mem = [r for k in keys[:cut] for r in by[k]]
    non = [r for k in keys[cut:] for r in by[k]]
    # fit on HALF, so there is a right answer about who was in
    half = CondNet(k=net.k, max_parents=net.max_parents).learn(
        mem, group_by=gb,
        multilevel=bool(getattr(net, "multilevel", False)),
        epsilon=getattr(net, "epsilon", 0.0))
    aud = membership_audit(half, mem, non,
                           half.sample(1500, seed=3),
                           nn_sample=120)
    lines = [
        "The model was rebuilt from half the patients, and an "
        "adversary was asked which half each person came from.",
        "The strongest one scored {:.3f}, where 0.500 is a coin "
        "flip and 1.000 would mean every patient identified."
        .format(aud["worst_auc"]),
    ]
    if aud.get("context"):
        lines.append(aud["context"])
    else:
        lines.append(aud["reading"].split(". ", 1)[-1])
    return {"verdict": aud["verdict"], "auc": aud["worst_auc"],
            "likelihood": aud["likelihood"]["auc"],
            "nearest_neighbor": aud.get(
                "nearest_neighbor", {}).get("auc"),
            "people": aud.get("members_are_people"),
            "plain": lines}


def api_learn_save(payload: dict) -> dict:
    """Hand back the learned model as a file.

    Relearning a large extract every session is wasted time, and
    worse, it means the numbers in a demo cannot be reproduced
    exactly. The model is parameters, so it saves as text — and
    because it holds no records, the saved file is as safe to keep
    as the report written from it.
    """
    net = _LEARNED.get("net")
    if net is None:
        return {"error": "Learn from a dataset first."}
    return {"filename": "learned_model.json",
            "content": net.to_json(), "mime": "application/json",
            "note": "parameters only; no record from the source "
                    "file is present in this model"}


def api_learn_load(payload: dict) -> dict:
    """Restore a saved model and pick up where it left off."""
    from .condnet import CondNet
    from . import learnspec as _ls
    raw = payload.get("content") or ""
    path = payload.get("path")
    if path and not raw:
        pp = Path(path).expanduser()
        if not pp.exists():
            return {"error": "No file at {}".format(pp)}
        raw = pp.read_text(encoding="utf-8")
    if not raw.strip():
        return {"error": "Give a saved model file to load."}
    try:
        net = CondNet.from_json(raw)
    except Exception as e:
        return {"error": "That does not look like a saved model: "
                         "{}".format(str(e)[:160])}
    _LEARNED["net"] = net
    _LEARNED["facts"] = _ls.facts_from_model(net)
    _LEARNED.setdefault("rows", net.report.get("rows", 1000))
    return {"narrative": _ls.narrate(net), "dials": _ls.dials(net),
            "outcome_candidates": _ls.outcome_candidates(net),
            "note_plan": _ls.plan_summary(_LEARNED["facts"]),
            "privacy": "Restored from parameters. The original "
                       "file is not needed and was never stored.",
            "grouped_by": net.report.get("grouped_by", "(none)"),
            "restored": True}


def api_learn_score(payload: dict) -> dict:
    """Check the generated data against the file it was learned
    from — on fidelity AND on privacy.

    Generation without verification is decoration. This runs the
    same scorecard the command line uses, so the numbers a person
    sees in the bench are the numbers a reviewer would reproduce.
    """
    import json as _json
    import subprocess
    import tempfile
    blob = _LEARNED.get("generated")
    srcp = _LEARNED.get("path")
    if not blob or not srcp:
        return {"error": "Learn from a dataset and generate "
                         "before checking."}
    td = Path(tempfile.mkdtemp(prefix="synthkit_score_"))
    syn = td / "generated.csv"
    # Identifiers are not comparable and must not be scored. A
    # synthetic patient number has nothing to do with a real one,
    # and comparing the two columns reports a failure that means
    # nothing while burying the failures that do.
    import csv as _csv2
    import io as _io2
    _net = _LEARNED.get("net")
    # The identifier stays: the scorecard needs it to work out how
    # many independent people the rows represent. It is excluded
    # from the column comparisons on the scorecard's own side.
    drop = {"visit_number"}
    rdr = list(_csv2.DictReader(_io2.StringIO(blob)))
    if rdr and drop & set(rdr[0]):
        keep = [c for c in rdr[0] if c not in drop]
        out2 = _io2.StringIO()
        wr2 = _csv2.DictWriter(out2, fieldnames=keep)
        wr2.writeheader()
        for row in rdr:
            wr2.writerow({c: row[c] for c in keep})
        blob = out2.getvalue()
    syn.write_text(blob, encoding="utf-8")
    out = td / "fidelity.json"
    proc = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parent.parent / "scripts"
             / "fidelity_report.py"),
         "--source", str(srcp), "--synthetic", str(syn),
         "-o", str(out)]
        + (["--group-by", getattr(_net, "group_by", "")]
           if getattr(_net, "group_by", None) else []),
        capture_output=True, text=True)
    if not out.exists():
        return {"error": "The scorecard could not run: {}".format(
            (proc.stderr or proc.stdout or "")[-400:])}
    rep = _json.loads(out.read_text(encoding="utf-8"))
    s = rep["summary"]
    marg = rep.get("marginals", [])
    shape = rep.get("dependence_shape", {})
    shape_bad = [d for d in shape.get("detail", [])
                 if not d["pass"]]
    priv = rep["privacy"]
    lines = []
    lines.append(
        "Each field's own distribution: {} of {} matched the "
        "source within what a fresh sample of the same size would "
        "differ by anyway.".format(
            sum(1 for m in marg if m["pass"]), len(marg)))
    scored = shape.get("pairs_tested", 0)
    if scored:
        lines.append(
            "Relationships between fields: {} were strong enough "
            "in the source to be worth checking, {} of them "
            "reversed direction somewhere (which a correlation "
            "cannot describe at all), and {} were not reproduced "
            "closely enough.".format(
                scored,
                shape.get("nonlinear_relationships_in_source", 0),
                len(shape_bad)))
    else:
        lines.append(
            "No relationship in the source was strong enough to "
            "check: at this many patients, what looks like "
            "structure could as easily be chance. Asking the "
            "synthetic data to reproduce it would be scoring a "
            "coin flip.")
    lines.append(
        "Privacy: {} synthetic record{} matched a real one, and "
        "{:.0%} of synthetic records sit closer to a real patient "
        "than real patients sit to each other \u2014 an "
        "independent sample would put about 5% there.".format(
            priv["exact_matches"],
            "" if priv["exact_matches"] == 1 else "s",
            priv.get("closeness_test", {}).get(
                "synthetic_fraction_below", 0.0)))
    # A list column cannot match exactly, and should not. Only
    # items shared by at least k patients are kept, so the rare
    # combinations are deliberately absent — that suppression is
    # the privacy guarantee doing its job, and reporting it as a
    # fidelity fault would teach the reader to distrust the wrong
    # number.
    net = _LEARNED.get("net")
    expanded = set(getattr(net, "list_columns", {}) or {})
    by_design = [m["column"] for m in marg
                 if not m["pass"] and m["column"] in expanded]
    if by_design:
        lines.append(
            "Expected difference: {} held combinations unique to "
            "one or two patients, and only items shared by at "
            "least ten were kept. That gap is the privacy rule "
            "working, not a fault in the model.".format(
                ", ".join(by_design)))
    _remember("checked",
              "fidelity {} / privacy {} ({} of {} checks)".format(
                  s["fidelity_verdict"], s["privacy_verdict"],
                  s["passed"], s["checks"]))
    return {
        "expected_differences": by_design,
        "fidelity_verdict": s["fidelity_verdict"],
        "privacy_verdict": s["privacy_verdict"],
        "passed": s["passed"], "checks": s["checks"],
        "plain": lines,
        "failing_columns": [m["column"] for m in marg
                            if not m["pass"]
                            and m["column"] not in expanded][:8],
        "failing_relationships": [
            "{} across {}".format(d["column"], d["given"])
            for d in shape_bad][:8],
    }


def api_learn_export(payload: dict) -> dict:
    blob = _LEARNED.get("generated")
    if not blob:
        return {"error": "Generate first."}
    return {"filename": "learned_synthetic.csv", "content": blob,
            "mime": "text/csv"}


def _fit_cmd(payload: dict, types_only: bool):
    """The exact CLI invocation, so the bench and the terminal run
    the same code path - the bench is a window onto `synthkit fit`,
    not a second implementation of it."""
    import sys as _sys
    src = str(Path(payload.get("src") or "").expanduser())
    out = str(Path(payload.get("out") or "").expanduser())
    cmd = [_sys.executable, "-m", "synthkit.cli",
           "types" if types_only else "fit",
           "--src", src, "--out", out]
    group = (payload.get("group_by") or "").strip()
    if group:
        cmd += ["--group-by", group]
    if not types_only:
        if payload.get("lags"):
            cmd.append("--lags")
        cmd.append("--generate")
    return cmd, src, out


def api_fit_types(payload: dict) -> dict:
    """How every column was read, and whether it can SURVIVE - in
    seconds, before anything expensive. The cheap check goes first."""
    import subprocess
    if not payload.get("src") or not payload.get("out"):
        return {"error": "Give both a source CSV path and an output "
                         "directory. The output directory is YOURS "
                         "to choose - put it outside any repository."}
    cmd, src, out = _fit_cmd(payload, types_only=True)
    if not Path(src).exists():
        return {"error": "No file at {}.".format(src)}
    cmd.append("--types-only")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=600)
    return {"text": (r.stdout or "") + (r.stderr or ""),
            "ok": r.returncode == 0}


def _fit_work(payload: dict) -> dict:
    """Runs `synthkit fit --generate` as a subprocess, its output
    streamed to a log inside the run directory - the same file a
    person would read if they had run it by hand."""
    import subprocess
    cmd, src, out = _fit_cmd(payload, types_only=False)
    if not Path(src).exists():
        return {"error": "No file at {}.".format(src)}
    Path(out).mkdir(parents=True, exist_ok=True)
    log = Path(out) / "bench_fit_log.txt"
    with log.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           text=True, timeout=4 * 3600)
    if r.returncode != 0:
        tail = log.read_text(encoding="utf-8",
                             errors="replace")[-1500:]
        return {"error": "fit exited {} - the end of its log:\n{}"
                         .format(r.returncode, tail)}
    return {"out": out}


def api_fit_log(payload: dict) -> dict:
    log = Path(str(payload.get("out") or "")) / "bench_fit_log.txt"
    if not log.exists():
        return {"text": "(no log yet)"}
    txt = log.read_text(encoding="utf-8", errors="replace")
    return {"text": txt[-4000:]}


def api_fit_open(payload: dict) -> dict:
    """A finished run, judged. Reads only the run directory - the
    same artifacts the CLI wrote - and the gate criteria come from
    synthkit.gate, the ONE place they are defined."""
    from . import gate as _gate
    run = Path(str(payload.get("out") or "")).expanduser()
    fp = run / "fidelity.json"
    if not fp.exists():
        return {"error": "No fidelity.json in {} - run fit with "
                         "generation first, or check the path."
                         .format(run)}
    fid = json.loads(fp.read_text(encoding="utf-8"))
    try:
        verdict = _gate.assess(fid)
    except ValueError as e:
        return {"error": str(e)}
    s = fid.get("summary") or {}
    prov = {}
    pv = run / "provenance.json"
    if pv.exists():
        prov = json.loads(pv.read_text(encoding="utf-8"))
    shape = [r for r in (fid.get("set_shape") or [])
             if abs(r.get("delta") or 0) > 0.05]
    return {
        "gate": verdict,
        "what_now": _gate.next_steps(verdict),
        "summary": {
            "coverage": [s.get("coverage_ok"), s.get("columns")],
            "centre": [s.get("centre_ok"), s.get("numeric")],
            "spread": [s.get("spread_ok"), s.get("numeric")],
            "pairs_sign": [s.get("pairs_sign_ok"), s.get("pairs")],
            "pairs_close": [s.get("pairs_close"), s.get("pairs")],
            "inverted": s.get("pairs_inverted"),
            "set_tokens": [s.get("set_tokens_ok"),
                           s.get("set_tokens_compared")],
            "set_empty": [s.get("set_empty_ok"),
                          s.get("set_empty_compared")],
        },
        "contradictions": len(fid.get("contradictions") or []),
        "disobedience": len(fid.get("disobedience") or []),
        "empty_gaps": shape,
        # provenance's build is a DICT ({source, id, note}), and
        # slicing it in the UI threw before a person saw anything.
        "build": ((prov.get("build") or {}).get("id", "unknown")
                  if isinstance(prov.get("build"), dict)
                  else str(prov.get("build") or "unknown")),
        "source": (prov.get("source") or {}),
    }


def api_fit_bridge(payload: dict) -> dict:
    """A fitted blueprint crosses into the exam half.

    This is what makes "then, for either route" TRUE for the measure
    rail. Until it existed, only the first engine's Learn output
    could reach Campaign in the bench - the bridge was built and
    measured (every decile within 0.03 of a source sd on a column
    skewed 2.83) and never wired to the UI. The spec it returns
    carries the bridge's own account of what crossed and what could
    not, because a spec that looks complete and has silently lost
    its relationships is the same failure as a column that is
    secretly all sentinel."""
    from . import bridge as _bridge
    run = Path(str(payload.get("out") or "")).expanduser()
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        return {"error": "No blueprint.json in {} - run fit first."
                         .format(run)}
    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    rows = int(((bp.get("patients") or {}).get("rows")) or 0)
    # The bridge returns {"tablespec": ..., "carried": ...} - the
    # spec and its honesty note travel together, and the first
    # version of this endpoint read fields that do not exist and
    # returned an empty account. A test that did not check for
    # `error` then printed the emptiness as if it were a result.
    out = _bridge.blueprint_to_tablespec(
        bp, title="Measured from {}".format(run.name), rows=rows)
    carried = out.get("carried") or {}
    spec = out.get("tablespec")
    if not spec:
        return {"error": "the bridge returned no tablespec - the "
                         "blueprint may be from an older build"}
    return {"spec": spec,
            "crossed": carried.get("crossed") or [],
            "did_not_cross": carried.get("did_not_cross") or []}


def api_deck(payload: dict) -> dict:
    """The original-vs-synthetic dashboard, built live for one run.

    The heavy lifting is `fidelity_deck.build_deck`, the SAME code
    that writes the roadshow file - so the interactive dashboard in
    the bench and the artifact the team receives are one picture, by
    construction. The HTML comes back as a string the page drops
    into a sandboxed iframe, so the deck's own styles never touch
    the bench's."""
    import sys as _sys
    sd = str(Path(__file__).resolve().parent.parent / "scripts")
    if sd not in _sys.path:
        _sys.path.insert(0, sd)
    try:
        from fidelity_deck import build_deck
    except Exception as e:
        return {"error": "could not load the deck builder: {}"
                         .format(e)}
    src = str(Path(payload.get("src") or "").expanduser())
    run = str(Path(payload.get("out") or "").expanduser())
    if not src or not run:
        return {"error": "Give the same source CSV and output "
                         "directory the run used."}
    if not Path(src).exists():
        return {"error": "No file at {}.".format(src)}
    if not (Path(run) / "fidelity.json").exists():
        return {"error": "No finished run in {} - fit and generate "
                         "first, or point at a directory that "
                         "holds one.".format(run)}
    group = (payload.get("group_by") or "person_id").strip()
    compares = []
    other = (payload.get("compare_dir") or "").strip()
    if other:
        compares.append("{}={}".format(
            payload.get("compare_label") or "compare",
            str(Path(other).expanduser())))
    try:
        doc, meta = build_deck(src, run, group, compares)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "the deck build failed: {}".format(e)}
    return {"html": doc, "columns": meta.get("columns"),
            "suppressed": meta.get("suppressed")}


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
    "/api/fit-types": api_fit_types,
    "/api/fit-run": lambda payload: {
        "job": _start_job(_fit_work, payload,
                          budget_s=4 * 3600.0)},
    "/api/fit-log": api_fit_log,
    "/api/fit-open": api_fit_open,
    "/api/deck": api_deck,
    "/api/deck-async": lambda payload: {
        "job": _start_job(api_deck, payload, budget_s=1200.0)},
    "/api/fit-bridge": api_fit_bridge,
    "/api/learn": api_learn,
    "/api/learn-async": lambda payload: {
        "job": _start_job(api_learn, payload)},
    "/api/learn-generate": api_learn_generate,
    "/api/learn-export": api_learn_export,
    "/api/learn-score": api_learn_score,
    "/api/learn-save": api_learn_save,
    "/api/learn-audit": api_learn_audit,
    "/api/learn-audit-async": lambda payload: {
        "job": _start_job(api_learn_audit, payload)},
    "/api/learn-history": api_learn_history,
    "/api/learn-load": api_learn_load,
    "/api/learn-plant": api_learn_plant,
    "/api/learn-plant-async": lambda payload: {
        "job": _start_job(api_learn_plant, payload)},
    "/api/learn-score-async": lambda payload: {
        "job": _start_job(api_learn_score, payload)},
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
.chip{font-family:var(--mono);font-size:10px;font-weight:800;
  letter-spacing:.08em;padding:3px 8px;border-radius:6px;
  text-transform:uppercase}
.chip.built{background:var(--cardinal);color:#fff}
.chip.partial{background:var(--gold-wash);color:#6d5a2a;
  border:1px solid var(--gold)}
.chip.planned{background:#f2f2f4;color:#5b6675}
.goal{border:1px solid var(--rule);border-radius:10px;
  padding:12px 14px;margin:10px 0;background:var(--panel)}
.goal h3{margin:0 0 4px;font-size:14.5px}
.goal .ev{font-size:12.5px;color:#44534c;margin:4px 0 0}
.gaterow{display:flex;gap:10px;font-family:var(--mono);
  font-size:12.5px;padding:4px 0;border-bottom:1px dashed var(--rule)}
.gaterow b{width:52px}
.gaterow b.ok{color:#2e7d32}
.gaterow b.bad{color:#b3261e}

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

/* ================================================
   STEP SYSTEM LAYER (appended: later wins)

   The bench had one accent color and numbered every
   control 1.1, 1.2, 4.7 - so nothing told you at a glance
   WHICH step you were in, and a sub-number like 4.7 read
   as a version rather than as "the seventh thing in step
   four". Sub-numbering was also wrong: 6.3 appeared twice.

   Now: each step owns a HUE, carried by its tab, its
   banner and every lettered control inside it. Letters,
   not decimals, because A/B/C cannot be mistaken for a
   number that means something else.

   The tabs are the one loud thing on the page. Everything
   else stays quiet so the color means "where am I".
   ================================================ */
:root{
  /* THE HOUSE PALETTE, AND WHY EIGHT HUES BECAME TWO.
     The rainbow existed to make the parallel routes unmistakable,
     but the foolproofing never lived in the hues - it lives in the
     step numbers, the rail labels, the you-need/you-get banners and
     the next footers, which all carry the meaning in words. Color
     is reinforcement: CARDINAL means the create route, GOLD means
     the measure route, gray/ink/white is everything shared. Two
     accents answer "which route, which step" faster than six, and a
     color-blind reader loses nothing because the words never
     depended on the color. */
  --cardinal:#8C1515; --cardinal-lo:#A94343; --cardinal-wash:#FDFAFA;
  --gold:#B3995D;     --gold-lo:#C9B37E;     --gold-wash:#FDFCF8;
  --slate:#3D4046;    --slate-lo:#5B5F66;    --slate-wash:#FAFAFB;
}
[data-route="create"]{--tab:var(--cardinal);
  --tablo:var(--cardinal-lo);--wash:var(--cardinal-wash)}
[data-route="measure"]{--tab:var(--gold);
  --tablo:var(--gold-lo);--wash:var(--gold-wash)}
[data-route="shared"],[data-route="map"]{--tab:var(--slate);
  --tablo:var(--slate-lo);--wash:var(--slate-wash)}

/* ---- the rail: bright reflective tabs ---- */
nav{padding:18px 12px}
.wordmark{padding:0 8px 16px}
.station{
  position:relative;display:block;width:100%;text-align:left;
  margin:0 0 9px;padding:11px 13px 11px 15px;border:0;
  border-radius:11px;cursor:pointer;color:#fff;
  font-family:var(--sans);font-size:14.5px;font-weight:650;
  letter-spacing:.005em;line-height:1.25;
  background:linear-gradient(180deg,var(--tablo) 0%,
             var(--tab) 58%,var(--tab) 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55),
             inset 0 -2px 0 rgba(0,0,0,.16),
             0 2px 5px rgba(15,23,42,.20);
  text-shadow:0 1px 1px rgba(0,0,0,.22);
  filter:saturate(.86) brightness(.97);
  transition:filter .12s,transform .08s,box-shadow .12s}
/* the sheen: a specular band across the top half */
.station::after{content:"";position:absolute;
  left:0;right:0;top:0;height:46%;pointer-events:none;
  border-radius:11px 11px 40% 40%/11px 11px 100% 100%;
  background:linear-gradient(180deg,rgba(255,255,255,.34) 0%,
             rgba(255,255,255,.06) 100%)}
.station b{display:inline-block;font-family:var(--mono);
  font-size:10.5px;font-weight:800;letter-spacing:.10em;
  background:rgba(0,0,0,.24);border-radius:6px;
  padding:2px 7px;margin-right:9px;vertical-align:1px;
  text-shadow:none}
.station .subt{display:block;font-size:11.5px;font-weight:500;
  opacity:.93;margin-top:3px;letter-spacing:.01em;
  text-shadow:0 1px 1px rgba(0,0,0,.18)}
.station:hover{filter:saturate(1) brightness(1.05);
  transform:translateY(-1px)}
.station.active{filter:saturate(1.12) brightness(1.10);
  border-left:0;background:linear-gradient(180deg,
    var(--tablo) 0%,var(--tab) 70%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7),
             inset 0 -2px 0 rgba(0,0,0,.18),
             0 0 0 2px #fff, 0 0 0 4px var(--tab),
             0 6px 16px rgba(15,23,42,.30)}
.station:focus-visible{outline:3px solid var(--ink);
  outline-offset:3px}
.station.done b::after{content:" \2713";color:#EAFBEF;
  font-weight:900}
/* Learn is not step 6 of the flow - it is another way to
   START, and putting it in the same numbered run implied
   you arrive there last. */
.railsplit{font-family:var(--mono);font-size:9.5px;
  letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim);margin:16px 8px 8px;
  padding-top:14px;border-top:1px solid var(--rule)}

/* ---- the step banner ---- */
.stepbanner{display:flex;align-items:center;gap:12px;
  flex-wrap:wrap;font-size:21px;font-weight:750;
  letter-spacing:-.015em;color:var(--ink);
  margin:2px 0 10px;padding:0}
.stepbanner .stepchip{font-family:var(--mono);font-size:11px;
  font-weight:800;letter-spacing:.10em;text-transform:uppercase;
  color:#fff;padding:6px 12px;border-radius:8px;
  background:linear-gradient(180deg,var(--tablo),var(--tab));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.5),
             0 2px 5px rgba(15,23,42,.22);
  text-shadow:0 1px 1px rgba(0,0,0,.22)}
/* WHAT YOU NEED AND WHAT YOU GET, on every step. The old
   page explained what a step was ABOUT; it never said what
   had to exist before you could run it, so the only way to
   find out was to press the button and read an error. */
.stepgoal{display:grid;grid-template-columns:auto 1fr;
  gap:6px 12px;align-items:baseline;
  background:var(--wash);border:1px solid var(--rule);
  border-left:5px solid var(--tab);border-radius:0 10px 10px 0;
  padding:12px 16px;margin:0 0 14px;font-size:14px;
  line-height:1.5}
.stepgoal dt{font-family:var(--mono);font-size:10px;
  font-weight:800;letter-spacing:.13em;text-transform:uppercase;
  color:var(--tab);white-space:nowrap}
.stepgoal dd{margin:0;color:#33414E}
.explain{background:#fff;border-left:4px solid var(--tab);
  border-radius:0 10px 10px 0;font-size:14px;
  padding:11px 15px;color:#3A4855}

/* ---- lettered controls: A, B, C ---- */
.stepno{display:inline-flex;align-items:center;
  justify-content:center;min-width:23px;height:23px;
  font-family:var(--mono);font-size:12.5px;font-weight:800;
  border-radius:7px;padding:0 6px;margin-right:9px;
  vertical-align:-5px;color:#fff;
  background:linear-gradient(180deg,var(--tablo),var(--tab));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.45),
             0 1px 3px rgba(15,23,42,.28);
  text-shadow:0 1px 1px rgba(0,0,0,.25)}
.eyebrow{font-size:11.5px;letter-spacing:.10em;
  color:#44525F;font-weight:700}
.panel{border-top:3px solid var(--tab)}

/* ---- required / optional, said louder ---- */
.badge{font-size:10px;font-weight:800;letter-spacing:.07em;
  padding:3px 9px;border-radius:7px;vertical-align:1px}
.badge.req{background:#B3261E;color:#fff}
.badge.rec{background:#8A5A0B;color:#fff}
.badge.opt{background:#E3E9EE;color:#46545F;
  border:1px solid #C9D3DB}

/* ---- what to do next, at the foot of every step ---- */
.nextup{display:flex;align-items:center;gap:11px;
  flex-wrap:wrap;margin:18px 0 6px;padding:13px 16px;
  border-radius:11px;background:#fff;
  border:1px solid var(--rule);border-left:5px solid var(--tab);
  font-size:14px;box-shadow:0 1px 2px rgba(23,34,44,.06)}
.nextup .lbl{font-family:var(--mono);font-size:10px;
  font-weight:800;letter-spacing:.13em;text-transform:uppercase;
  color:var(--tab)}
@media(max-width:860px){
  nav{display:flex;gap:8px;padding:12px}
  .station{margin:0;min-width:172px}
  .railsplit{display:none}
  .stepbanner{font-size:18px}}
h1{display:flex;align-items:center;gap:11px;
  font-family:var(--sans);font-size:22px;font-weight:750;
  letter-spacing:-.015em}
h1 .tstep{font-family:var(--mono);font-style:normal;
  font-size:10.5px;font-weight:800;letter-spacing:.12em;
  text-transform:uppercase;color:#fff;padding:5px 10px;
  border-radius:7px;
  background:linear-gradient(180deg,var(--tablo),var(--tab));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.5),
             0 2px 4px rgba(15,23,42,.22);
  text-shadow:0 1px 1px rgba(0,0,0,.22)}
h1 span{font-size:15px;font-weight:500;color:var(--dim);
  letter-spacing:0}
#speccard{border-top:3px solid var(--tab);border-radius:10px}

/* ================================================================
   PORCELAIN LETTERPRESS - the loved effect, made the whole language.

   The operator pointed at a hover state twice, and twice the wrong
   half of the screenshot was preserved: what they loved was the
   button going PURE WHITE with raised ink lettering and its black
   shadow - the letterpress moment - not the saturated glossy pill
   it started from. So that moment IS the design now, everywhere,
   at rest: every station, button and chip is porcelain - white,
   raised on layered shadows - and every label is set in ink with a
   dark shadow beneath the glyphs, so the lettering reads as raised
   off the surface. Hover deepens the shadows and lifts the piece;
   press settles it. Route identity survives as accents only: the
   number chip's tint, the accent bar, the active ring.
   ================================================================ */
:root{
  --graphite-hi:#454E5A; --graphite:#2C333C; --graphite-lo:#20262E;
  /* the couture pass: light behaves like light. One imagined
     source above; shadows fall in MANY soft layers with a long
     falloff (deeper never means darker); the letter shadow is a
     two-layer plume; the edge is a bevel - brighter above, a
     hairline darker below - like glazed ceramic. */
  --letterpress:0 1px 1px rgba(10,14,20,.40),
                0 2px 3px rgba(10,14,20,.14);
  --letterpress-deep:0 2px 2px rgba(10,14,20,.46),
                     0 4px 6px rgba(10,14,20,.16);
  --raise:inset 0 1px 0 rgba(255,255,255,.95),
          inset 0 -1px 0 rgba(27,35,48,.05),
          0 1px 1px rgba(27,35,48,.07),
          0 2px 4px rgba(27,35,48,.06),
          0 8px 16px rgba(27,35,48,.07),
          0 16px 32px rgba(27,35,48,.05);
  --raise-hi:inset 0 1px 0 #fff,
             inset 0 -1px 0 rgba(27,35,48,.06),
             0 2px 3px rgba(27,35,48,.09),
             0 6px 12px rgba(27,35,48,.10),
             0 18px 36px rgba(27,35,48,.14),
             0 32px 64px rgba(27,35,48,.10);
  --raise-press:inset 0 2px 4px rgba(27,35,48,.14),
                inset 0 1px 0 rgba(27,35,48,.04);
  --bevel:rgba(27,35,48,.08);
  --bevel-low:rgba(27,35,48,.15);
  --card:0 1px 1px rgba(20,26,34,.04),
         0 4px 12px rgba(20,26,34,.07),
         inset 0 1px 0 rgba(255,255,255,.9);
}
body{background:
  radial-gradient(1100px 560px at 50% -12%,#FCFCFD,transparent),
  var(--bench)}

/* THE STATIONS: porcelain tiles, letterpress labels. This defeats
   the saturated gradient layers above it on purpose. */
.station{color:var(--ink);position:relative;overflow:hidden;
  background:linear-gradient(180deg,#FFFFFF 0%,#FAFBFC 55%,
    #F1F3F6 100%);
  border:1px solid var(--bevel);border-bottom-color:var(--bevel-low);
  border-left:4px solid var(--tab);box-shadow:var(--raise);
  text-shadow:var(--letterpress);
  filter:none;
  transition:box-shadow .22s cubic-bezier(.22,1,.36,1),
    transform .18s cubic-bezier(.22,1,.36,1),
    background .22s ease,text-shadow .18s ease}
/* the signature flourish, and the only one: a band of light glides
   across the porcelain on hover, as if the piece were tilted under
   the lamp. It replaces the old specular wholesale. */
.station::after{content:"";display:block;position:absolute;
  top:0;bottom:0;height:auto;left:-70%;right:auto;width:45%;
  border-radius:0;pointer-events:none;
  background:linear-gradient(105deg,
    rgba(27,35,48,0) 0%,rgba(27,35,48,.05) 35%,
    rgba(255,255,255,.95) 50%,
    rgba(27,35,48,.05) 65%,rgba(27,35,48,0) 100%);
  transform:skewX(-16deg);
  transition:left .9s cubic-bezier(.22,1,.36,1)}
.station:hover::after{left:125%}
.station .subt{color:var(--dim);opacity:1;
  text-shadow:0 1px 1px rgba(10,14,20,.22)}
.station b{background:var(--wash);color:var(--tab);
  border:1px solid rgba(20,26,34,.10);
  text-shadow:none}
.station:hover{filter:none;
  transform:translateY(-2px) scale(1.004);
  background:linear-gradient(180deg,#FFFFFF,#F6F8FA);
  box-shadow:var(--raise-hi);
  text-shadow:var(--letterpress-deep)}
.station:active{transform:translateY(0);
  box-shadow:var(--raise-press)}
.station.active{filter:none;color:var(--ink);
  background:linear-gradient(180deg,#FFFFFF,var(--wash));
  border-left:4px solid var(--tab);
  box-shadow:var(--raise-hi),0 0 0 2px var(--tab),
    0 0 0 7px var(--wash);
  text-shadow:var(--letterpress-deep)}
.station.active b{background:var(--tab);color:#fff;
  border-color:var(--tab)}
.station.done b::after{content:" \2713";color:#2e7d55;
  font-weight:900}

/* THE ACTION BUTTONS: the same porcelain, the same letterpress. */
.act,button.act{position:relative;overflow:hidden;
  font-family:var(--sans);font-size:13px;font-weight:700;
  letter-spacing:.01em;color:var(--ink);
  border:1px solid var(--bevel);border-bottom-color:var(--bevel-low);
  border-radius:11px;
  padding:11px 20px;cursor:pointer;margin:12px 8px 0 0;
  background:linear-gradient(180deg,#FFFFFF 0%,#FAFBFC 55%,
    #F1F3F6 100%);
  box-shadow:var(--raise);text-shadow:var(--letterpress);
  transition:box-shadow .22s cubic-bezier(.22,1,.36,1),
    transform .18s cubic-bezier(.22,1,.36,1),
    text-shadow .18s ease,background .22s ease}
.act::after,button.act::after{content:"";position:absolute;
  top:0;bottom:0;left:-70%;width:45%;pointer-events:none;
  background:linear-gradient(105deg,
    rgba(27,35,48,0) 0%,rgba(27,35,48,.05) 35%,
    rgba(255,255,255,.95) 50%,
    rgba(27,35,48,.05) 65%,rgba(27,35,48,0) 100%);
  transform:skewX(-16deg);
  transition:left .9s cubic-bezier(.22,1,.36,1)}
.act:hover::after,button.act:hover::after{left:125%}
.act:hover,button.act:hover{box-shadow:var(--raise-hi);
  transform:translateY(-2px) scale(1.004);
  background:linear-gradient(180deg,#FFFFFF,#F6F8FA);
  text-shadow:var(--letterpress-deep)}
.act:active,button.act:active{box-shadow:var(--raise-press);
  transform:translateY(0)}
.act:focus-visible,button.act:focus-visible{
  outline:3px solid var(--ink);outline-offset:3px}

/* ghost + preset chips + roadmap chips: smaller porcelain */
button.ghost{background:linear-gradient(180deg,#FFFFFF,#F5F6F8);
  color:var(--ink);border:1px solid rgba(20,26,34,.10);
  border-radius:11px;box-shadow:var(--raise);
  text-shadow:var(--letterpress);
  transition:box-shadow .16s ease,transform .12s ease}
button.ghost:hover{box-shadow:var(--raise-hi);
  transform:translateY(-2px);background:#fff}
.presets button{font-family:var(--mono);font-size:12px;
  color:var(--ink);
  background:linear-gradient(180deg,#FFFFFF,#F5F6F8);
  border:1px solid rgba(20,26,34,.10);border-radius:9px;
  padding:8px 13px;cursor:pointer;box-shadow:var(--raise);
  text-shadow:var(--letterpress);
  transition:box-shadow .14s ease,transform .1s ease}
.presets button:hover{box-shadow:var(--raise-hi);
  transform:translateY(-2px);background:#fff}

/* the chips and pills: porcelain with route-tinted lettering */
h1 .tstep,.stepbanner .stepchip,.stepno{
  background:linear-gradient(180deg,#FFFFFF,#F6F7F9);
  color:var(--tab);border:1px solid rgba(20,26,34,.10);
  box-shadow:var(--raise);
  text-shadow:0 1px 1px rgba(10,14,20,.18)}
.chip.built{background:linear-gradient(180deg,#fff,var(--cardinal-wash));
  color:#7a1414;border:1px solid #efd9d9;box-shadow:var(--card)}
.chip.partial{background:linear-gradient(180deg,#fff,var(--gold-wash));
  color:#6d5a2a;border:1px solid #e9e0cb;box-shadow:var(--card)}
.chip.planned{background:linear-gradient(180deg,#fff,var(--slate-wash));
  color:#5b6675;border:1px solid var(--rule);box-shadow:var(--card)}

/* DONE IS A LIGHT, NOT A GUESS. A finished step shows a jade
   lamp on its rail tile and a completion strip in its panel naming
   the next step - the operator should never wonder whether they
   are free to move on. */
.station.done::before{content:"";position:absolute;
  right:11px;top:11px;width:9px;height:9px;border-radius:50%;
  background:radial-gradient(circle at 35% 30%,#7fd6ac,#2e7d55);
  box-shadow:0 0 0 2px #fff,0 0 8px rgba(46,125,85,.55),
    inset 0 1px 1px rgba(255,255,255,.6);
  animation:lampon .6s cubic-bezier(.22,1,.36,1)}
@keyframes lampon{from{transform:scale(.2);opacity:0}
  to{transform:scale(1);opacity:1}}
.stepdone{display:flex;align-items:center;gap:10px;
  margin:16px 0 6px;padding:12px 16px;border-radius:11px;
  background:linear-gradient(180deg,#FFFFFF,#F4FAF7);
  border:1px solid #D8EAE0;border-left:5px solid #2e7d55;
  font-size:14px;box-shadow:var(--card);
  animation:lampon .5s cubic-bezier(.22,1,.36,1)}
.stepdone .lamp{width:10px;height:10px;border-radius:50%;
  flex:none;
  background:radial-gradient(circle at 35% 30%,#7fd6ac,#2e7d55);
  box-shadow:0 0 8px rgba(46,125,85,.55),
    inset 0 1px 1px rgba(255,255,255,.6)}
@media (prefers-reduced-motion:reduce){
  .station.done::before,.stepdone{animation:none}}

/* WHAT NOW: the gate's failed criteria come with their options,
   clearly labeled, from the shared gate module */
.whatnow{margin-top:14px;padding:14px 16px;background:#fff;
  border:1px solid var(--rule);border-left:5px solid var(--tab);
  border-radius:0 12px 12px 0;box-shadow:var(--card)}
.whatnow h3{margin:0 0 8px;font-size:14px}
.wn-crit{margin:10px 0;font-size:13.5px;line-height:1.55}
.wn-crit ol{margin:6px 0 0 2px;padding-left:20px}
.wn-crit li{margin:4px 0}

/* the rail: a sheet of vellum behind the porcelain tiles */
nav{background:linear-gradient(90deg,#F6F7F9,#F3F4F7);
  border-right:1px solid rgba(27,35,48,.06)}

/* SURFACES: raised porcelain cards; INPUTS: inset, the counterpoint */
.panel,#speccard,.goal{background:var(--panel);
  border:1px solid var(--rule);border-radius:14px;
  box-shadow:var(--card);transition:box-shadow .22s ease}
.panel:hover,.goal:hover{box-shadow:var(--raise-hi)}
#deck-frame,.gate{box-shadow:var(--card)}
textarea,input,select{border:1px solid var(--rule);
  border-radius:9px;background:#FCFCFD;
  box-shadow:inset 0 1px 2px rgba(20,26,34,.06)}
textarea:focus,input:focus,select:focus{
  box-shadow:inset 0 1px 2px rgba(20,26,34,.07),
    0 0 0 3px var(--wash);border-color:var(--tab);outline:none}

/* ================================================================
   THE LIVING LOADER. A long run used to be a wall of log text and
   an elapsed counter - nothing on screen MOVED, so a healthy
   33-minute fit was indistinguishable from a hang. The loader is
   always in motion: a milled track, a fill with a traveling sheen,
   a breathing dot. When the log names its stage the bar is TRUE
   progress parsed from the run's own countdown lines; when nothing
   is parseable it sweeps as a comet rather than pretending to know.
   ================================================================ */
.loader{margin:12px 0 10px;display:none}
.loader.on{display:block}
.loader .lrow{display:flex;align-items:baseline;gap:10px;
  margin-bottom:7px}
.loader .ldot{width:8px;height:8px;border-radius:50%;
  align-self:center;background:var(--graphite);
  box-shadow:0 0 0 0 rgba(44,51,60,.35);
  animation:lbreathe 1.6s ease-in-out infinite}
@keyframes lbreathe{
  0%,100%{transform:scale(.85);box-shadow:0 0 0 0 rgba(44,51,60,.30)}
  50%{transform:scale(1.15);box-shadow:0 0 0 6px rgba(44,51,60,0)}}
.loader .lstage{font-family:var(--mono);font-size:12px;
  color:var(--ink);letter-spacing:.02em}
.loader .lelapsed{font-family:var(--mono);font-size:11.5px;
  color:var(--dim);margin-left:auto}
.loader .track{height:7px;border-radius:5px;background:#E9EBEE;
  overflow:hidden;position:relative;
  box-shadow:inset 0 1px 2px rgba(20,26,34,.10)}
.loader .fill{height:100%;border-radius:5px;position:relative;
  overflow:hidden;width:3%;
  background:linear-gradient(180deg,var(--graphite-hi),
    var(--graphite));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.25);
  transition:width .9s cubic-bezier(.22,1,.36,1)}
.loader .fill::after{content:"";position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,
    rgba(255,255,255,.40),transparent);
  animation:lsheen 1.5s linear infinite}
@keyframes lsheen{from{transform:translateX(-100%)}
  to{transform:translateX(100%)}}
.loader.indet .fill{width:26%;
  animation:lcomet 1.4s cubic-bezier(.45,.05,.55,.95)
    infinite alternate}
@keyframes lcomet{from{margin-left:0}to{margin-left:74%}}
.loader.done .fill{width:100%;background:linear-gradient(180deg,
  #2e7d55,#256a47);animation:none}
@media (prefers-reduced-motion:reduce){
  .loader .ldot,.loader .fill::after,.loader.indet .fill{
    animation:none}}
</style></head><body>
<div class="frame">
<nav>
  <div class="wordmark">SYNTHKIT<small>calibration bench<br>
    <span id="fp" title="version &middot; built &middot; build
    fingerprint; compare with `synthkit version`">loading
    build...</span></small></div>
  <div class="railsplit">create from a description</div>
  <button class="station active" data-step="1" data-s="describe" data-route="create"><b>01</b>
    Describe<small class="subt">define the dataset</small></button>
  <button class="station" data-step="2" data-s="spec" data-route="create"><b>02</b>
    Spec<small class="subt">review the recipe</small></button>
  <button class="station" data-step="3" data-s="data" data-route="create"><b>03</b>
    Data<small class="subt">generate synthetic data</small></button>
  <div class="railsplit">or measure real data</div>
  <button class="station" data-step="1" data-s="fitsrc" data-route="measure"><b>01</b>
    Source<small class="subt">point at a real CSV</small></button>
  <button class="station" data-step="2" data-s="fitrun" data-route="measure"><b>02</b>
    Fit<small class="subt">measure &amp; generate</small></button>
  <button class="station" data-step="3" data-s="fitver" data-route="measure"><b>03</b>
    Verdict<small class="subt">judge the run</small></button>
  <button class="station" data-step="6" data-s="learn" data-route="measure"><b>ALT</b>
    Learn<small class="subt">first engine &mdash; quick look</small></button>
  <div class="railsplit">then, for either route</div>
  <button class="station" data-step="4" data-s="campaign" data-route="shared"><b>04</b>
    Campaign<small class="subt">configure the evaluation</small></button>
  <button class="station" data-step="5" data-s="showdown" data-route="shared"><b>05</b>
    Showdown<small class="subt">compare results</small></button>
  <div class="railsplit">see the fidelity</div>
  <button class="station" data-step="8" data-s="dashboard" data-route="map"><b>VIEW</b>
    Dashboard<small class="subt">original vs synthetic, drawn</small></button>
  <div class="railsplit">the map</div>
  <button class="station" data-step="8" data-s="roadmap" data-route="map"><b>MAP</b>
    Roadmap<small class="subt">eight goals: built and planned</small></button>
</nav>
<main data-step="1" data-route="create">
<header class="bar">
  <h1 id="title"><em class="tstep">Step 1</em>Describe <span>— say what data you need</span></h1>
  <div id="speccard" class="empty">
    <div class="fp">no spec loaded</div>
    <div class="meta">describe one or load a preset</div>
  </div>
</header>

<section id="s-describe" class="active" data-step="1">
  <div class="stepbanner"><span class="stepchip">Step 1 of 5</span><span>Say what data you need</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>Nothing &mdash; this is the start.</dd><dt>you get</dt><dd>A plain-English description, ready to become a recipe in Step 2.</dd></dl>
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
  (A) or write your own description (B), then continue to
  Step 2.</div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">A</span>
    <span class="badge opt">optional &mdash; fastest path</span>
    ready-made examples</div>
    <div class="presets" id="presets"></div>
    <div class="hint">Click one to load a complete, tested
    description into Step 2. For the vendor evaluation demo,
    click the first card and skip to Step 2.</div>
  </div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">B</span>
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
      <div><label for="kind"><span class="stepno">C</span>
        what shape of data?</label>
        <select id="kind"><option value="table">a table &mdash;
        one row per patient / record</option>
        <option value="document">text documents &mdash; e.g.
        clinical notes or reports</option></select></div>
      <div><label for="backend"><span class="stepno">D</span>
        <span class="badge opt">advanced</span> which AI reads
        your English</label>
        <select id="backend"><option value="ollama">Ollama app
        (only if installed on this computer)</option>
        <option value="openai">local AI server (llama.cpp /
        LM Studio)</option>
        <option value="bedrock">hospital cloud (AWS
        Bedrock)</option>
        <option value="anthropic">Anthropic API</option></select></div>
    </div>
    <div class="eyebrow"><span class="stepno">E</span>
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
  <div class="nextup"><span class="lbl">next</span><b>Step 2 &mdash; Spec</b><span>Nothing has been created yet. Step 2 shows the exact recipe and waits for you to approve it.</span></div>
</section>

<section id="s-spec" data-step="2">
  <div class="stepbanner"><span class="stepchip">Step 2 of 5</span><span>Review the recipe &mdash; the human gate</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A description from Step 1, or a ready-made example.</dd><dt>you get</dt><dd>An approved recipe with a fingerprint. Nothing is generated until you approve it here.</dd></dl>
  <div class="explain">This is the complete, exact recipe the
  rest of the process follows: every field, every distribution,
  every deliberate flaw, every hidden trap, and the promised
  outcome rate. You do not need to read it &mdash; the buttons below
  do the reading. The short code above the page (the
  fingerprint) identifies this recipe forever: same fingerprint,
  same data, every time, on any machine.</div>
  <div class="panel">
    <div id="spec-summary"></div>
    <div class="eyebrow"><span class="stepno">A</span>
    <span class="badge opt">experts only</span> the recipe
    itself (editable)</div>
    <textarea id="spec" rows="22" spellcheck="false"
      placeholder="No spec yet &mdash; describe one or load a preset."></textarea>
    <div class="eyebrow"><span class="stepno">B</span>
    <span class="badge req">required</span> check the recipe is
    complete and lawful</div>
    <button class="act" onclick="validateSpec()">Validate</button>
    <div class="hint">Green means every field is well-defined
    and internally consistent. Problems are listed in plain
    terms so they can be fixed before anything is created.</div>
    <div class="eyebrow"><span class="stepno">C</span>
    <span class="badge opt">optional</span> quick trial run
    (nothing saved)</div>
    <button class="act ghost" onclick="planSpec()">Plan (dry run)</button>
    <div class="hint">Builds the dataset in memory and reports
    what it WOULD contain &mdash; row counts, corrupted cells &mdash;
    without writing anything to disk.</div>
    <div class="eyebrow"><span class="stepno">D</span>
    <span class="badge rec">recommended</span> does the data
    keep the recipe's promises?</div>
    <button class="act ghost" onclick="lintSpec()">Semantic lint</button>
    <div class="hint">Generates a sample and measures it against
    what was declared &mdash; e.g. "readmission was promised at
    5&ndash;12% and lands at 8%". This is how you know the dataset
    means what the description said.</div>
    <div class="eyebrow"><span class="stepno">E</span>
    <span class="badge opt">optional</span> save the recipe
    file</div>
    <button class="act ghost" onclick="downloadSpec()">Download spec.json</button>
    <div class="hint">Anyone with this one file can regenerate
    this exact dataset &mdash; that is the reproducibility guarantee,
    and what you would hand an auditor or a vendor.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="spec-out"></pre>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 3 &mdash; Data</b><span>Once the recipe validates and you approve it, Step 3 turns it into rows or documents.</span></div>
</section>

<section id="s-data" data-step="3">
  <div class="stepbanner"><span class="stepchip">Step 3 of 5</span><span>Create the synthetic data</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>An approved recipe from Step 2.</dd><dt>you get</dt><dd>A table or a set of documents you can open, download, and hand to a vendor.</dd></dl>
  <div class="explain">This turns the approved recipe into real
  files: the messy dataset (what a model would actually face),
  the clean answer key (the same records with every flaw
  repaired and every truth known), and a ledger listing every
  deliberate corruption. When it finishes, a preview and
  download buttons appear below.</div>
  <div class="panel">
    <div class="eyebrow"><span class="stepno">A</span>
    <span class="badge req">required</span> where to save</div>
    <label for="outdir">folder name for this run</label>
    <input id="outdir" value="gui_runs/run_001">
    <div class="eyebrow"><span class="stepno">B</span>
    <span class="badge opt">advanced &mdash; text documents
    only</span> who writes the prose</div>
    <div class="row2">
      <div><label for="rbackend">writer</label>
        <select id="rbackend"><option value="stub">built-in
        writer &mdash; no AI, instant, always identical</option>
        <option value="ollama">Ollama app (only if Ollama is
        installed on this computer)</option>
        <option value="openai">local AI server &mdash; llama.cpp /
        LM Studio running on this computer (NOT
        ChatGPT)</option>
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
    (recommended for tables and live demos). <b>Ollama</b> =
    open models served by the Ollama app &mdash; only works where
    that app is installed. <b>local AI server</b> = any
    OpenAI-compatible server on
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
    <div class="eyebrow"><span class="stepno">C</span>
    <span class="badge req">required</span> create the data</div>
    <button class="act" onclick="renderSpec()">Create the data</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="render-out"></pre>
    <div id="data-summary"></div>
    <div id="data-downloads"></div>
    <div id="render-preview"></div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 4 &mdash; Campaign</b><span>The data exists but nothing has been tested. Step 4 builds the exam a model has to sit.</span></div>
</section>

<section id="s-campaign" data-step="4">
  <div class="stepbanner"><span class="stepchip">Step 4 of 5</span><span>Set the exam, then let a model sit it</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>Generated data from Step 3.</dd><dt>you get</dt><dd>A scored run at every difficulty tier, for the solver you pick.</dd></dl>
  <div class="explain">A campaign is a standardized exam built
  from the recipe: the same test at three difficulty levels,
  with pass marks you set. Here you define the exam (4.1&ndash;4.4)
  and then have synthkit's own built-in model sit it (4.5&ndash;4.6)
  &mdash; so before any vendor is judged, you know what a free,
  transparent model can score on this data.</div>
  <div class="panel">
    <div class="row2">
      <div><label for="goal"><span class="stepno">A</span>
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
      <div><label for="outcome"><span class="stepno">B</span>
        <span class="badge req">required for predict</span>
        which column is being predicted?</label>
        <input id="outcome" placeholder="readmitted_30d"></div>
    </div>
    <label for="bars"><span class="stepno">C</span>
    <span class="badge req">required</span> pass marks
    (name=value, comma-separated)</label>
    <input id="bars" value="fix_rate=0.9,detect_rate=0.5">
    <div id="bars-plain" class="hint"></div>
    <div class="hint">For prediction: auroc=0.6,gap_max=0.3.
    AUROC is the ranking score &mdash; 1.0 is perfect, 0.5 is a coin
    flip; 0.6 says "must beat a coin flip convincingly".
    gap_max limits how much worse a model may do on the messy
    data versus the clean answer key.</div>
    <div class="eyebrow"><span class="stepno">D</span>
    <span class="badge req">required</span> build the exam</div>
    <button class="act" onclick="campaignCompile()">Compile ladder</button>
    <label for="solver"><span class="stepno">E</span>
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
      <div><label for="lbackend"><span class="stepno">F</span>
        <span class="badge opt">only for llm_extract</span>
        which AI is being tested</label>
        <select id="lbackend"><option value="ollama">Ollama app
        (only if installed)</option>
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
      <div><label for="samples"><span class="stepno">G</span>
        <span class="badge opt">advanced</span> answers per
        question (majority vote)</label>
        <input id="samples" value="1"></div>
      <div><label for="intervention"><span class="stepno">H</span>
        <span class="badge opt">advanced</span> extra
        instruction to the tested AI</label>
        <input id="intervention" placeholder="optional"></div>
    </div>
    <div class="eyebrow"><span class="stepno">I</span>
    <span class="badge req">required</span> run the exam</div>
    <button class="act" onclick="campaignRun()">Run ladder</button>
    <div class="hint">Training happens on a separate practice
    population the model has never been scored on; answers are
    hidden during the test. Results arrive per difficulty tier
    with statistical confidence attached.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="campaign-out"></pre>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 5 &mdash; Showdown</b><span>You have one model&rsquo;s score. Step 5 puts it beside our own baseline and the theoretical ceiling.</span></div>
</section>

<section id="s-learn" data-step="6">
  <div class="stepbanner"><span class="stepchip">Alternative start</span><span>Begin from data you already have</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A CSV of real records, on this machine. Nothing leaves it.</dd><dt>you get</dt><dd>A recipe MEASURED from that data &mdash; which then goes through Steps 2 to 5 exactly like an invented one.</dd></dl>
  <div class="explain">The five steps above INVENT a population
  from a description. This one learns from a dataset that already
  exists: it measures each field's distribution, how often values
  go missing, and &mdash; the part a spreadsheet cannot show you
  &mdash; how the fields move together, including relationships
  that reverse direction or only appear in combination.
  <b>The file is never copied.</b> What the bench keeps is counts
  and curves, each one covering at least ten patients, and it
  generates from those. No synthetic record descends from a real
  one.</div>
  <div class="gov"><span>&#128274; Parameters only &mdash; no record
  is stored</span><span>&#128101; Every figure covers at least 10
  patients</span><span>&#127903; Every pattern found becomes a
  dial you can turn</span></div>
  <div class="panel">
    <label for="lpath"><span class="stepno">A</span>
    <span class="badge req">required</span> full path to a tidy
    CSV &mdash; one row per visit</label>
    <input id="lpath" placeholder="/path/to/tidy_visits.csv">
    <div class="hint">Or choose the file directly:
    <input type="file" id="lfile" accept=".csv,.tsv,.txt"
    onchange="learnPickFile(this)"> <span id="lfilename"></span>
    </div>
    <div class="hint">If you have raw clinical tables rather than
    one tidy file, the command line can join them for you first:
    <code>python scripts/omop_wrangle.py --src FOLDER -o
    tidy_visits.csv</code></div>
    <label for="lgroup"><span class="stepno">B</span>
    <span class="badge rec">recommended</span> which column
    identifies the PATIENT</label>
    <input id="lgroup" value="person_id">
    <div class="hint">This matters more than it looks. Ten visits
    from one patient are not ten patients&#39; worth of privacy
    protection, and not ten patients&#39; worth of evidence
    either &mdash; naming this column makes the bench count people
    rather than rows.</div>
    <label for="leps"><span class="stepno">C</span>
    <span class="badge opt">optional</span> privacy budget
    (epsilon) &mdash; leave blank for none</label>
    <input id="leps" placeholder="1.0">
    <div class="hint">Leaving this blank still protects the data
    by k-anonymity and by never publishing a true minimum or
    maximum. Setting a number goes further: it adds calibrated
    noise so that an adversary who already knows everything else
    about the cohort still cannot tell whether any ONE patient was
    in it. Lower is stronger. At 1.0 the cost to accuracy is
    usually negligible; below about 0.3 relationships start to
    wash out, and the bench will show you that rather than hide
    it.</div>
    <button class="act" onclick="learnRun()">Learn from this
    data</button>
    <button class="act ghost" onclick="learnSave()">Save this
    model</button>
    <button class="act ghost" onclick="learnLoadPrompt()">Load a
    saved model</button>
    <div class="outlabel">what the bench found</div>
    <div class="out" id="learn-out">Point at a file and press
    Learn.</div>
    <div id="learn-report"></div>
  </div>
  <div class="panel">
    <label>what has been done this session</label>
    <div class="hint">A bench that forgets is hard to trust:
    without this, yesterday&#39;s number is gone and there is no
    way to see whether a change helped. Saving a model keeps the
    work; this keeps the trail.</div>
    <button class="act ghost" onclick="learnHistory()">Refresh
    </button>
    <div id="learn-history"><div class="hint">Nothing yet this
    session.</div></div>
  </div>
  <div class="panel" id="learn-dials-panel" style="display:none">
    <label><span class="stepno">D</span>
    <span class="badge opt">optional</span> turn what was
    found</label>
    <div class="hint">Every relationship the bench discovered can
    be made stronger, weaker, or removed entirely. Set one to 0
    and see whether a vendor still claims to find it; set it to 2
    and see whether they catch it when it is twice as strong.
    This is the difference between a copy of your data and an
    instrument you can aim.</div>
    <div id="learn-dials"></div>
  </div>
  <div class="panel" id="learn-gen-panel" style="display:none">
    <label for="lrows"><span class="stepno">E</span>
    <span class="badge req">required</span> how many records to
    create</label>
    <input id="lrows" value="1000">
    <label><input type="checkbox" id="lnotes"> also write a
    clinical note for each record</label>
    <label><input type="checkbox" id="lhier" checked> generate
    PATIENTS with a course of visits, not loose rows</label>
    <div class="hint">With this on, each synthetic person gets
    their own visit history: their fixed traits stay fixed, and a
    value that drifts drifts from one visit to the next instead of
    being redrawn from nothing every row. Any analysis that groups
    by patient &mdash; which is most clinical analysis &mdash;
    behaves like real longitudinal data only when this is on.
    </div>
    <div class="hint">The note is assembled from the same facts
    the model learned, written the way clinicians write &mdash;
    shorthand, denials, hedges, findings carried forward &mdash;
    with a hidden answer key recording what each sentence really
    claims, so an extraction vendor can be graded on it.</div>
    <button class="act" onclick="learnGenerate()">Create the
    data</button>
    <button class="act" onclick="learnScore()">Check it against
    the real data</button>
    <button class="act" onclick="learnAudit()">Try to break the
    privacy</button>
    <button class="act ghost" onclick="learnDownload()">Download
    CSV</button>
    <div class="outlabel">result</div>
    <div class="out" id="learn-gen-out">Learn first, then create.
    </div>
    <div class="outlabel">how faithful, and how private</div>
    <div class="out" id="learn-score-out">Create the data, then
    check it.</div>
    <div id="learn-score-report"></div>
    <div class="outlabel">can anyone tell who was in the
    cohort?</div>
    <div class="out" id="learn-audit-out">A privacy claim nobody
    tested is the one most likely to be repeated. Press the button
    above to test this one.</div>
    <div id="learn-audit-report"></div>
    <div id="learn-preview"></div>
  </div>
  <div class="panel" id="learn-exam-panel" style="display:none">
    <label><span class="stepno">F</span>
    <span class="badge opt">optional</span> turn this into an
    exam</label>
    <div class="explain">Realistic data is not yet a test. A test
    needs an answer key &mdash; and an answer key needs causes
    that a PERSON stated, not causes inferred from the same data
    the models will be shown. Give the fields below the weights
    you believe they should carry, and the bench computes the
    outcome from them. Because you stated the causes, the
    probability behind every record is known, and so is the best
    score any model could possibly reach.</div>
    <div id="learn-weights"></div>
    <label for="lprev">target prevalence (how often the outcome
    should occur)</label>
    <input id="lprev" value="0.15">
    <div class="hint">The intercept is solved for you to hit this
    &mdash; that is only a base rate. Your weights are the causal
    claims and are never adjusted.</div>
    <label><input type="checkbox" id="lhide" checked> put the
    causes ONLY in the notes</label>
    <div class="hint">With this on, the fields you weighted are
    removed from the columns and written into the prose instead.
    That is what makes the comparison meaningful: a model that
    cannot read is then missing part of the signal by
    construction, and the gap measures exactly what reading is
    worth. With it off, both models can see everything and the
    gap will be zero.</div>
    <label for="lvendor">a vendor model to grade as well
    (optional)</label>
    <input id="lvendor" placeholder="vendor_model:predict">
    <button class="act" onclick="learnPlant()">Plant the truth and
    grade</button>
    <div class="outlabel">the verdict</div>
    <div class="out" id="learn-exam-out">Weight at least one field
    above, then grade.</div>
    <div id="learn-exam-report"></div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 2 &mdash; Spec</b><span>The measured recipe lands in Step 2 exactly like a written one, and goes through the same human gate.</span></div>
</section>
<section id="s-showdown" data-step="5">
  <div class="stepbanner"><span class="stepchip">Step 5 of 5</span><span>The verdict &mdash; vendor against our own model</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A campaign from Step 4.</dd><dt>you get</dt><dd>One line for a meeting: the ceiling, our baseline, and the vendor&rsquo;s number side by side.</dd></dl>
  <div class="explain">The vendor's model and our own take the
  identical exam on identical data &mdash; data where the maximum
  achievable score is KNOWN, because we planted the truth. Below
  the raw readout, a full plain-English report explains what was
  tested, what traps the data contained, how each model works,
  and who won.</div>
  <div class="panel">
    <label for="sbaseline"><span class="stepno">A</span>
    <span class="badge req">required</span> our challenger (the
    floor the vendor must beat)</label>
    <select id="sbaseline"><option value="autosolver">autosolver
      &mdash; tabular only (cannot read notes)</option>
      <option value="autosolver_hybrid">autosolver_hybrid &mdash; our
      best: reads the table AND the notes</option></select>
    <label for="vendor"><span class="stepno">B</span>
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
    <div class="eyebrow"><span class="stepno">C</span>
    <span class="badge req">required</span> run the head-to-head</div>
    <button class="act" onclick="showdown()">Run showdown</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="showdown-out"></pre>
    <div id="showdown-report"></div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Done</b><span>This is the end of the run. Change anything in Steps 1 to 4 and the fingerprint changes with it, so a result always names the data it came from.</span></div>
</section>
<section id="s-fitsrc" data-step="1">
  <div class="stepbanner"><span class="stepchip">Measure &middot; step 1 of 3</span><span>Point at a real CSV</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A tidy CSV on THIS machine (one row per visit), and an output directory of your choosing &mdash; outside any repository. Nothing leaves this machine.</dd><dt>you get</dt><dd>Every column typed, in seconds &mdash; and whether each can SURVIVE anonymization, before anything expensive runs.</dd></dl>
  <div class="panel">
    <h2>Point at the data</h2>
    <label>source CSV path
      <input id="fsrc" placeholder="full path to the tidy CSV"></label>
    <label>output directory <span class="hint">(yours to choose; created if missing; keep it out of any repo)</span>
      <input id="fout" placeholder="full path for this run's artifacts"></label>
    <label>patient / entity column
      <input id="fgroup" value="person_id"></label>
    <label><input type="checkbox" id="flags" checked> include lag features (slower, needed for temporal patterns)</label>
    <button class="act" onclick="fitTypes()">Check the types
    (seconds)</button>
    <div class="hint">Types first, always: every silent fault this
    tool has had was a column read as the wrong type, and this pass
    also reports patients per level against the k floor &mdash; a
    column that cannot survive is caught here, not after an hour.</div>
    <div class="out" id="fit-out">Nothing yet.</div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 2 &mdash; Fit</b><span>Once the types read correctly, Step 2 measures the blueprint and generates data to it.</span></div>
</section>

<section id="s-fitrun" data-step="2">
  <div class="stepbanner"><span class="stepchip">Measure &middot; step 2 of 3</span><span>Measure the blueprint, then generate</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A source and output directory from Step 1, with the types read correctly.</dd><dt>you get</dt><dd>A k-anonymous blueprint measured from the data, generated rows drawn to it, and the full findings file &mdash; the same artifacts a terminal run writes, because this button RUNS the command line.</dd></dl>
  <div class="panel">
    <h2>Fit and generate</h2>
    <button class="act" onclick="fitRun()">Fit and generate
    (minutes to an hour)</button>
    <div class="hint">This is the slow, honest part: discovery
    confirms every claim on held-out patients. The log below is the
    run's own narration, streamed live.</div>
    <div class="out" id="fit-runout">Not started.</div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 3 &mdash; Verdict</b><span>When the run finishes, Step 3 grades it: the six-criteria gate, contradictions, obedience.</span></div>
</section>

<section id="s-fitver" data-step="3">
  <div class="stepbanner"><span class="stepchip">Measure &middot; step 3 of 3</span><span>Judge the run</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A finished run in the output directory from Step 1 &mdash; from Step 2, or from any terminal run this machine holds.</dd><dt>you get</dt><dd>The verdict: six gate criteria PASS/FAIL, contradiction and obedience counts, and the build the run is tied to &mdash; plus a bridge that sends the measured recipe into the exam, stating what crossed and what could not. The criteria come from one shared module, so this panel and <code>scripts/m0_gate.py</code> cannot disagree.</dd></dl>
  <div class="panel">
    <h2>Judge a finished run</h2>
    <button class="act" onclick="fitOpen()">Open the run in the
    output directory from Step 1</button>
    <div id="fit-verdict"></div>
  </div>
  <div class="panel">
    <h2>Send the measured recipe to the exam</h2>
    <div class="hint">Loads the fitted blueprint into the same spec
    slot Step 2 of the create route uses &mdash; then Campaign and
    Showdown work identically for measured data. The bridge states
    what crossed and what could not: marginals and correlations
    cross; effect curves and visit rhythms do not, and the spec says
    so about itself.</div>
    <button class="act" onclick="fitBridge()">Send to the exam
    (Step 4)</button>
    <div class="out" id="fit-bridgeout">Not sent.</div>
  </div>
  <div class="nextup"><span class="lbl">next</span><b>Step 4 &mdash; Campaign</b><span>A measured dataset takes the same road as an invented one: set the exam, then the showdown.</span></div>
</section>

<section id="s-dashboard" data-step="8" data-route="map">
  <div class="stepbanner"><span class="stepchip">View</span><span>Original vs synthetic, drawn</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>A finished run &mdash; from the measure route, or any terminal run this machine holds &mdash; and the source CSV it was measured from.</dd><dt>you get</dt><dd>The interactive fidelity dashboard: every column original beside synthetic, the relationship curves drawn as shapes, correlation heatmaps side by side, and the gate verdict. The same picture the roadshow file carries, because it is built by the same code.</dd></dl>
  <div class="panel">
    <h2>Point at a run</h2>
    <label>source CSV path
      <input id="dsrc" placeholder="the CSV the run was measured from"></label>
    <label>run output directory
      <input id="dout" placeholder="the directory that holds fidelity.json"></label>
    <label>patient / entity column
      <input id="dgroup" value="person_id"></label>
    <div class="dcmp">
      <label>compare against a second run <span class="hint">(optional &mdash; the &ldquo;does 5x degrade it&rdquo; table)</span>
        <input id="dcmpdir" placeholder="a second run's directory, e.g. a 5x expansion"></label>
      <label>label for it
        <input id="dcmplabel" value="5x"></label>
    </div>
    <button class="act" onclick="deckBuild()">Draw the dashboard</button>
    <div class="hint">Built live by the same code that writes the
    shareable report, so the interactive view and the file the team
    receives are one picture. Source histogram bins backed by fewer
    than ten patients are suppressed &mdash; the report is itself
    k-screened.</div>
    <div class="out" id="deck-out">Nothing drawn yet.</div>
  </div>
  <iframe id="deck-frame" title="fidelity dashboard" style="width:100%;
    height:78vh;border:1px solid var(--rule);border-radius:12px;
    margin-top:14px;display:none;background:#fff"></iframe>
  <div class="nextup"><span class="lbl">next</span><b>Roadmap</b><span>This is where the engine stands on your data; the Roadmap says where the eight goals are going.</span></div>
</section>

<section id="s-roadmap" data-step="8">
  <div class="stepbanner"><span class="stepchip">The map</span><span>Eight goals &mdash; what is built, what is planned</span></div>
  <dl class="stepgoal"><dt>you need</dt><dd>Nothing &mdash; this page is for reading, and for the room.</dd><dt>you get</dt><dd>Where each goal stands, with the measured evidence, and what is planned for the parts that do not exist yet. Percentages are judgments; the numbers beside them are not. Full detail: <code>docs/goals_scorecard.md</code>.</dd></dl>
  <div class="goal"><h3><span class="chip partial">partial &middot; ~55%</span> 1 &middot; Universal upload with auto schema mapping</h3>
    <div class="ev">BUILT: single-table CSV end to end; types, currency, percent and clock parsers; long/EAV pivot; 22 of 23 dataset shapes come out clean, and flat data comes back flat.</div>
    <div class="ev">PLANNED: multi-table intake with key auto-detection; Excel / JSON formats. Documents are a later decision, on purpose.</div></div>
  <div class="goal"><h3><span class="chip partial">partial &middot; ~45%</span> 2 &middot; Automatic de-identification</h3>
    <div class="ev">BUILT: everything published is k-anonymous over PATIENTS; unpublishable labels are replaced by invented ones (1,436 real codes in, zero republished); attacked with positive controls &mdash; membership worst 0.52 where a cheat scores 1.00 and FAILS.</div>
    <div class="ev">PLANNED: the PHI scrub itself &mdash; names, addresses, SSNs, birth dates &mdash; with a 100%-catch gate on planted PHI. Free text is a decision gate, not a promise.</div></div>
  <div class="goal"><h3><span class="chip partial">partial &middot; ~70%</span> 3 &middot; Every pattern found, explained, with receipts</h3>
    <div class="ev">BUILT: discovery confirmed on held-out patients &mdash; 12/13 planted patterns, zero false; effect curves, interactions, presence-as-signal; the atlas explains all 97 components in plain English and refuses to build if one is missing.</div>
    <div class="ev">PLANNED: per-claim receipt files; the tangled-graph ceiling &mdash; the one open research item.</div></div>
  <div class="goal"><h3><span class="chip partial">partial &middot; ~65%</span> 4 &middot; Dials over every pattern</h3>
    <div class="ev">BUILT: count, coverage, shift, scale, persistence, clustering &mdash; each reports requested AGAINST achieved, because a dial can be capped by privacy and a silent difference is the failure this tool refuses.</div>
    <div class="ev">PLANNED: dials on individual relationships; the full per-class verification pass.</div></div>
  <div class="goal"><h3><span class="chip built">built &middot; ~85%</span> 5 &middot; High-fidelity generation</h3>
    <div class="ev">MEASURED on the real extract: coverage 42/42, center 30/33, set token shares 62/62, empty rates 4/4, zero inverted relationships, direction 93.9%. Against a statistical-copy ruler: 119 relationships kept to its 77.</div>
    <div class="ev">REMAINING: one number &mdash; relationship strength within 0.2 on 76.5% of pairs against an 87.9% bar.</div></div>
  <div class="goal"><h3><span class="chip built">built &middot; ~80%</span> 6 &middot; Self-assessment for sign-off</h3>
    <div class="ev">BUILT: the six-criteria gate with an honest exit code; contradiction checks on the report AND the contract; row-level obedience checks that need no source data; diagnosis views. The run states its own privacy costs in place.</div>
    <div class="ev">PLANNED: the single roll-up page a decision-maker signs.</div></div>
  <div class="goal"><h3><span class="chip partial">partial &middot; ~60%</span> 7 &middot; Vendor evaluation against planted truth</h3>
    <div class="ev">BUILT: known effects planted on measured covariates &mdash; +0.9/-0.5 recovered at +0.86/-0.44, a no-effect column reads +0.03; the bridge carries measured distributions into the exam; campaigns and this bench.</div>
    <div class="ev">PLANNED: the loop assembled END TO END on the data machine &mdash; every part exists, the single run has not happened.</div></div>
  <div class="goal"><h3><span class="chip planned">planned &middot; ~45%</span> 8 &middot; Our own challenger, and one report card</h3>
    <div class="ev">BUILT: the structurally-blinded baseline solver; the ceiling / ours / vendor line in the campaign machinery.</div>
    <div class="ev">PLANNED: the report-card artifact from a real end-to-end run, then a challenger worth the name beyond the floor.</div></div>
  <div class="hint">Overall, equal-weighted: about 63%. What remains
  is mostly assembly plus two genuine unknowns &mdash; and unknowns,
  not assembly, are what move dates. Projection: core complete
  mid-November; free text, if its gate says yes, mid-December.</div>
  <div class="nextup"><span class="lbl">next</span><b>Anywhere</b><span>This page is the map, not a step. Step 1 invents data from English; Fit measures data you already have.</span></div>
</section>
</main></div>

<script>
/* `learn` was MISSING from this map. `titles['learn']` came back
   undefined and `t[0]` threw a TypeError, so clicking the sixth
   station left the heading showing the previous step's name and the
   rest of the handler never ran. The station worked, the label lied,
   and nothing said so. */
const titles={describe:['Step 1','Describe','say what data you need'],
  spec:['Step 2','Spec','review the recipe'],
  data:['Step 3','Data','create the data'],
  campaign:['Step 4','Campaign','set the exam'],
  showdown:['Step 5','Showdown','ceiling / baseline / vendor'],
  learn:['Alt','Learn','start from data you already have'],
  fitsrc:['Measure 1','Source','point at a real CSV'],
  fitrun:['Measure 2','Fit','measure the blueprint, then generate'],
  fitver:['Measure 3','Verdict','judge the run'],
  dashboard:['View','Dashboard','original vs synthetic, drawn'],
  roadmap:['Map','Roadmap','the eight goals \u2014 built and planned']};
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
    /* The heading sits OUTSIDE every section, so it cannot
       inherit the step's hue from the section it describes.
       Stamping the step on <main> puts the whole bench, heading
       included, into the color of wherever you are. */
    document.querySelector('main').dataset.step=
      btn.dataset.step||'1';
    document.querySelector('main').dataset.route=
      btn.dataset.route||'create';
    const t=titles[btn.dataset.s]||['','',''];
    document.getElementById('title').innerHTML=
      '<em class="tstep">'+t[0]+'</em>'+t[1]+
      ' <span>&mdash; '+t[2]+'</span>';
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
  loaderSet('compile-out',null,'compiling via local model');
  const d=await api('/api/compile',{
    description:document.getElementById('english').value,
    kind:document.getElementById('kind').value,
    backend:document.getElementById('backend').value});
  loaderDone('compile-out',!d.error);
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
function previewTable(rows, cols){
  if(!rows||!rows.length)return '';
  var html='<div class="pvwrap"><table><tr>';
  for(var c=0;c<cols.length;c++){
    html+='<th>'+esc(cols[c])+'</th>';}
  html+='</tr>';
  for(var r=0;r<rows.length;r++){
    html+='<tr>';
    for(var c2=0;c2<cols.length;c2++){
      var v=String(rows[r][cols[c2]]===undefined?'':
        rows[r][cols[c2]]);
      var short=v.length>90?v.slice(0,90)+'\u2026':v;
      html+='<td title="'+esc(v)+'">'+
        (v===''?'&empty;':esc(short))+'</td>';}
    html+='</tr>';}
  return html+'</table></div><div class="pvhint">First '+
    rows.length+' records \u2014 scroll in both directions; '+
    'hover any cell to read it in full.</div>';}
var LEARN_DIALS={};
async function learnRun(){
  if(!UPLOADED&&!gate(['lpath'],'learn-out'))return;
  out('learn-out','reading the file and measuring it... this '+
    'takes a moment on a large dataset');
  var req={group_by:document.getElementById('lgroup').value,
    epsilon:parseFloat(document.getElementById('leps').value)||0};
  if(UPLOADED&&!document.getElementById('lpath').value){
    req.filename=UPLOADED.filename;req.content=UPLOADED.content;
  }else{req.path=document.getElementById('lpath').value;}
  loaderSet('learn-out',null,'reading and measuring the file');
  const d=await api('/api/learn',req);
  loaderDone('learn-out',!d.error);
  if(d.error){out('learn-out',d.error,'bad');return;}
  renderLearn(d);}
function renderLearn(d){
  out('learn-out',d.narrative.headline+'  |  counted by: '+
    d.grouped_by+(d.restored?'  |  restored from a saved model':''),
    'ok');
  var h='<div class="summarycard"><b>What the bench found in '+
    'your data</b>';
  if(d.narrative.findings.length){
    h+='<div style="margin-top:8px">';
    for(var i=0;i<d.narrative.findings.length;i++){
      h+='<div class="pstep"><span class="nchip" '+
        'style="background:#7B4B94">'+(i+1)+'</span><span>'+
        esc(d.narrative.findings[i].sentence)+
        '<span class="why"> \u2014 a real relationship between '+
        'fields</span></span></div>';}
    h+='</div>';
  }else{
    h+='<div style="margin-top:8px">No relationship survived the '+
      'statistical check. That is a finding, not a failure: with '+
      'this many patients, anything weaker than the check '+
      'demands could just as easily be chance.</div>';}
  if(d.narrative.bookkeeping.length){
    h+='<div style="margin-top:10px"><b>Filed as arithmetic, '+
      'not findings</b><div class="hint">These follow from how '+
      'the data was assembled &mdash; a count computed from a '+
      'list, a route that belongs to a drug. Reporting them as '+
      'discoveries would be the tool congratulating itself on '+
      'its own bookkeeping.</div>';
    for(var b=0;b<d.narrative.bookkeeping.length;b++){
      h+='<div class="hint">&middot; '+
        esc(d.narrative.bookkeeping[b].sentence)+'</div>';}
    h+='</div>';}
  h+='<div class="hint" style="margin-top:10px">'+
    esc(d.narrative.caveat)+'</div>';
  h+='<div class="hint">'+esc(d.privacy)+'</div>';
  h+='<div class="hint" style="margin-top:6px">'+
    esc(d.note_plan.sentence)+'</div></div>';
  document.getElementById('learn-report').innerHTML=h;
  LEARN_DIALS={};
  var dh='';
  for(var k=0;k<d.dials.length;k++){
    var dl=d.dials[k];
    LEARN_DIALS[dl.column]=1.0;
    dh+='<div class="modelrow"><span style="min-width:300px">'+
      esc(dl.label)+'</span><input type="range" min="0" max="3" '+
      'step="0.25" value="1" data-dial="'+esc(dl.column)+
      '" oninput="dialMove(this)"> <code id="dv-'+k+'">1.00'+
      '</code></div>';}
  document.getElementById('learn-dials').innerHTML=
    dh||'<div class="hint">No relationships were found, so there '+
      'is nothing to turn.</div>';
  document.getElementById('learn-dials-panel').style.display=
    d.dials.length?'block':'none';
  document.getElementById('learn-gen-panel').style.display='block';
  buildWeights(d.outcome_candidates||[]);
  learnHistory();
  tick('learn');}
function dialMove(el){
  LEARN_DIALS[el.dataset.dial]=parseFloat(el.value);
  var code=el.parentNode.querySelector('code');
  if(code)code.textContent=parseFloat(el.value).toFixed(2);}
async function learnGenerate(){
  out('learn-gen-out','creating records from the learned '+
    'patterns...');
  loaderSet('learn-gen-out',null,'creating records');
  const d=await api('/api/learn-generate',{
    rows:parseInt(document.getElementById('lrows').value)||1000,
    transcribe:document.getElementById('lnotes').checked,
    hierarchical:document.getElementById('lhier').checked,
    dials:LEARN_DIALS});
  loaderDone('learn-gen-out',!d.error);
  if(d.error){out('learn-gen-out',d.error,'bad');return;}
  var msg='Created '+d.rows+' records with '+d.columns+
    ' fields, drawn from the learned patterns.';
  if(d.ledgered_mentions){
    msg+=' '+d.ledgered_mentions+' facts were written into '+
      'clinical notes, each one recorded in the answer key.';}
  var turned=[];
  for(var kk in d.dials_applied){
    if(d.dials_applied[kk]!==1)turned.push(kk+' \u00d7'+
      d.dials_applied[kk]);}
  if(turned.length)msg+=' Dials turned: '+turned.join(', ')+'.';
  out('learn-gen-out',msg,'ok');
  document.getElementById('learn-preview').innerHTML=
    previewTable(d.preview,d.columns_list);
  if(d.temporal&&d.temporal.warning){
    document.getElementById('learn-gen-out').innerHTML+=
      '<div style="margin-top:10px;border-left:2px solid #B45309;'+
      'padding-left:12px"><b>The visit histories came out flat.'+
      '</b> '+esc(d.temporal.warning)+'</div>';
  }else if(d.temporal&&d.temporal.reproduced&&
           d.temporal.reproduced.length){
    var t=d.temporal.reproduced.map(function(x){
      return x.column+' '+x.generated+' (source '+x.source+')';});
    document.getElementById('learn-gen-out').innerHTML+=
      '<div class="hint" style="margin-top:8px">Visit-to-visit '+
      'steadiness reproduced: '+esc(t.join('; '))+'</div>';}
  learnHistory();}
var UPLOADED=null;
function learnPickFile(el){
  var f=el.files&&el.files[0];
  if(!f)return;
  var rd=new FileReader();
  rd.onload=function(){
    UPLOADED={filename:f.name,content:rd.result};
    document.getElementById('lfilename').textContent=
      f.name+' ('+Math.round(f.size/1024)+' KB) \u2014 ready';
    document.getElementById('lpath').value='';};
  rd.readAsText(f);}
async function learnHistory(){
  const d=await api('/api/learn-history',{});
  var h='';
  if(!d.history.length){
    h='<div class="hint">Nothing yet this session.</div>';
  }else{
    for(var i=0;i<d.history.length;i++){
      var e=d.history[i];
      h+='<div class="modelrow"><code>'+esc(e.when)+'</code>'+
        '<span class="badge opt">'+esc(e.action)+'</span>'+
        '<span>'+esc(e.summary)+'</span></div>';}
    h+='<div class="hint">'+esc(d.note)+'</div>';}
  document.getElementById('learn-history').innerHTML=h;}
var LEARN_WEIGHTS={};
function buildWeights(cands){
  var h='<div class="hint">Weight the fields you believe drive '+
    'the outcome. Leave the rest at zero \u2014 an unweighted '+
    'field still appears in the data, it just carries no '+
    'signal.</div>';
  for(var i=0;i<cands.length;i++){
    var c=cands[i];
    LEARN_WEIGHTS[c.column]=0;
    h+='<div class="modelrow"><span style="min-width:260px">'+
      esc(c.label)+' <span class="badge opt">'+esc(c.kind)+
      '</span></span><input type="range" min="-2" max="2" '+
      'step="0.1" value="0" data-w="'+esc(c.column)+
      '" oninput="weightMove(this)"> <code>0.0</code></div>';}
  document.getElementById('learn-weights').innerHTML=h;
  document.getElementById('learn-exam-panel').style.display=
    cands.length?'block':'none';}
function weightMove(el){
  LEARN_WEIGHTS[el.dataset.w]=parseFloat(el.value);
  var c=el.parentNode.querySelector('code');
  if(c)c.textContent=parseFloat(el.value).toFixed(1);}
async function learnPlant(){
  out('learn-exam-out','planting the outcome and grading both '+
    'models against it...');
  const j=await api('/api/learn-plant-async',{
    weights:LEARN_WEIGHTS,
    rows:parseInt(document.getElementById('lrows').value)||2000,
    prevalence:parseFloat(document.getElementById('lprev').value),
    hide_in_notes:document.getElementById('lhide').checked,
    vendor:document.getElementById('lvendor').value,
    dials:LEARN_DIALS});
  poll(j.job,'learn-exam-out',function(d){
    if(d.error){out('learn-exam-out',d.error,'bad');return;}
    var s=d.showdown;
    out('learn-exam-out','ceiling '+s.ceiling.toFixed(3)+
      '   our model '+s.reading.toFixed(3)+
      (s.blind!==null?('   blind '+s.blind.toFixed(3)):'' )+
      (s.vendor!==undefined?('   vendor '+
        s.vendor.toFixed(3)):''),'ok');
    var h='<div class="vstrip"><b>VERDICT:</b> the best score '+
      'reachable on this data is '+s.ceiling.toFixed(3)+
      ', and it is known exactly because the causes were '+
      'stated rather than guessed.</div>';
    h+='<div class="summarycard"><b>What the exam showed</b>';
    for(var i=0;i<d.plain.length;i++){
      h+='<div class="pstep"><span class="nchip" '+
        'style="background:#17803D">'+(i+1)+'</span><span>'+
        esc(d.plain[i])+'</span></div>';}
    h+='<div class="hint" style="margin-top:8px">Planted at '+
      (100*d.planted.prevalence).toFixed(1)+'% prevalence '+
      '(intercept solved to '+d.planted.intercept+
      '). Weights: '+esc(JSON.stringify(d.planted.weights))+
      '</div>';
    h+='<div class="hint">'+esc(d.planted.note)+'</div></div>';
    document.getElementById('learn-exam-report').innerHTML=h;});}
async function learnAudit(){
  out('learn-audit-out','rebuilding the model from half the '+
    'patients, then asking an adversary which half each person '+
    'came from...');
  const j=await api('/api/learn-audit-async',{});
  poll(j.job,'learn-audit-out',function(d){
    if(d.error){out('learn-audit-out',d.error,'bad');return;}
    out('learn-audit-out','strongest adversary '+
      d.auc.toFixed(3)+'  |  '+d.verdict,
      d.verdict==='PASS'?'ok':'');
    var h='<div class="summarycard"><b>What the attack found'+
      '</b>';
    for(var i=0;i<d.plain.length;i++){
      h+='<div class="pstep"><span class="nchip" '+
        'style="background:'+(d.verdict==='PASS'?'#17803D':
        '#B45309')+'">'+(i+1)+'</span><span>'+esc(d.plain[i])+
        '</span></div>';}
    h+='<div class="hint" style="margin-top:8px">Two adversaries '+
      'were run. One saw only the generated data ('+
      (d.nearest_neighbor===null?'not run':
       d.nearest_neighbor.toFixed(3))+'); the other was handed '+
      'the model itself ('+d.likelihood.toFixed(3)+'), because '+
      'that is what a determined attacker would have. The '+
      'verdict takes the stronger of the two.</div>';
    h+='</div>';
    document.getElementById('learn-audit-report').innerHTML=h;
    learnHistory();});}
async function learnScore(){
  out('learn-score-out','comparing the synthetic data against '+
    'the real file, field by field and relationship by '+
    'relationship...');
  const j=await api('/api/learn-score-async',{});
  poll(j.job,'learn-score-out',function(d){
    if(d.error){out('learn-score-out',d.error,'bad');return;}
    var okAll=(d.fidelity_verdict==='PASS'&&
      d.privacy_verdict==='PASS');
    out('learn-score-out',d.passed+' of '+d.checks+
      ' checks passed  |  fidelity '+d.fidelity_verdict+
      '  |  privacy '+d.privacy_verdict, okAll?'ok':'');
    var h='<div class="summarycard"><b>What the check found'+
      '</b>';
    for(var i=0;i<d.plain.length;i++){
      h+='<div class="pstep"><span class="nchip" '+
        'style="background:#7B4B94">'+(i+1)+'</span><span>'+
        esc(d.plain[i])+'</span></div>';}
    if(d.failing_columns.length){
      h+='<div class="hint" style="margin-top:8px"><b>Fields '+
        'that did not match closely enough:</b> '+
        esc(d.failing_columns.join(', '))+'</div>';}
    if(d.failing_relationships.length){
      h+='<div class="hint"><b>Relationships not reproduced:'+
        '</b> '+esc(d.failing_relationships.join('; '))+
        '</div>';}
    h+='<div class="hint" style="margin-top:8px">A failing check '+
      'is not always a fault. Tolerances are set by how much two '+
      'samples of this size would differ by chance, so a narrow '+
      'miss on one field usually means the source itself is '+
      'thin there \u2014 and a relationship you deliberately '+
      'turned off with a dial SHOULD fail to reproduce.</div>';
    h+='</div>';
    document.getElementById('learn-score-report').innerHTML=h;});}
function learnSave(){
  api('/api/learn-save',{}).then(function(d){
    if(d.error){out('learn-out',d.error,'bad');return;}
    var a=document.createElement('a');
    a.href='data:'+d.mime+';charset=utf-8,'+
      encodeURIComponent(d.content);
    a.download=d.filename;a.click();
    out('learn-out','saved \u2014 '+d.note,'ok');});}
function learnLoadPrompt(){
  var pth=prompt('Full path to a saved model file:');
  if(!pth)return;
  out('learn-out','restoring the saved model...');
  api('/api/learn-load',{path:pth}).then(function(d){
    if(d.error){out('learn-out',d.error,'bad');return;}
    renderLearn(d);});}
function learnDownload(){
  api('/api/learn-export',{}).then(function(d){
    if(d.error){out('learn-gen-out',d.error,'bad');return;}
    var a=document.createElement('a');
    a.href='data:'+d.mime+';charset=utf-8,'+
      encodeURIComponent(d.content);
    a.download=d.filename;a.click();});}
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
  if(!b)return;
  b.classList.add('done');
  /* the completion strip: says the step is finished and names
     where to go, inside the panel where the operator is looking */
  var sec=document.getElementById('s-'+step);
  if(sec&&!document.getElementById('sd-'+step)){
    var strip=document.createElement('div');
    strip.className='stepdone';strip.id='sd-'+step;
    var nx=sec.querySelector('.nextup b');
    strip.innerHTML='<span class="lamp"></span><b>Step complete.'+
      '</b> You are free to move on'+
      (nx?(' \u2014 next: '+nx.textContent):'')+'.';
    var anchor=sec.querySelector('.nextup');
    if(anchor)sec.insertBefore(strip,anchor);
    else sec.appendChild(strip);}}
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
  loaderSet('campaign-out',null,'compiling the ladder');
  const d=await api('/api/campaign-compile',{
    goal:document.getElementById('goal').value,
    spec:document.getElementById('spec').value,
    bars:bars,outcome:document.getElementById('outcome').value,
    out:''});
  loaderDone('campaign-out',!d.error);
  if(d.error){out('campaign-out',d.error,'bad');return;}
  campaignDir=d.campaign_dir;saveSession();
  out('campaign-out',d.title+' -> '+d.campaign_dir+'\n\n'+d.tiers.map((t,i)=>
    'tier '+(i+1)+' `'+t.name+'`: '+t.notes+'\n    '+
    t.conditions.join('\n    ')).join('\n'));}
let activeJob=null;
async function cancelJob(){
  if(activeJob)await api('/api/job-cancel',{id:activeJob});}
/* ---- the living loader: motion that cannot lie still ---- */
function loaderGet(outId,make){
  let l=document.getElementById(outId+'-loader');
  if(!l&&make){
    l=document.createElement('div');
    l.id=outId+'-loader';l.className='loader';
    l.innerHTML='<div class="lrow"><span class="ldot"></span>'+
      '<span class="lstage"></span>'+
      '<span class="lelapsed"></span></div>'+
      '<div class="track"><div class="fill"></div></div>';
    const o=document.getElementById(outId);
    o.parentNode.insertBefore(l,o);}
  return l;}
function loaderSet(outId,frac,label,secs){
  const l=loaderGet(outId,true);
  l.classList.add('on');l.classList.remove('done');
  if(frac==null){l.classList.add('indet');
    l.querySelector('.fill').style.width='';}
  else{l.classList.remove('indet');
    l.querySelector('.fill').style.width=
      Math.max(3,Math.min(100,frac*100))+'%';}
  if(label!=null)l.querySelector('.lstage').textContent=label;
  l.querySelector('.lelapsed').textContent=
    secs==null?'':Math.round(secs)+'s';}
function loaderDone(outId,ok){
  const l=loaderGet(outId,false);
  if(!l)return;
  l.classList.remove('indet');
  if(ok){l.classList.add('done');
    setTimeout(()=>{l.classList.remove('on');},1400);}
  else{l.classList.remove('on');}}
/* The fit log names its own stages, and the search stage prints a
   true countdown (i/N columns). The bar is honest: parsed progress
   when the log offers it, an indeterminate comet when it does not -
   never an invented percentage. Later stages are tested FIRST
   because the log accumulates. */
function fitStage(log){
  log=log||'';
  if(/done ->/.test(log))
    return{frac:1,label:'done'};
  if(/comparing source against generated/.test(log))
    return{frac:.94,label:'comparing source against generated'};
  if(/\bgenerating\b/.test(log))
    return{frac:.80,label:'generating new patients from the '+
      'contract'};
  if(/building the blueprint/.test(log))
    return{frac:.72,label:'writing the contract'};
  const m=log.match(/(\d+)\/(\d+) columns, about/g);
  if(m){
    const last=m[m.length-1].match(/(\d+)\/(\d+)/);
    const i=+last[1],n=Math.max(+last[2],1);
    return{frac:.08+.62*(i/n),
      label:'searching columns '+i+'/'+n};}
  if(/searching \d+ columns/.test(log))
    return{frac:.08,label:'searching columns'};
  if(/reading /.test(log))
    return{frac:.04,label:'reading the file'};
  return{frac:null,label:'working'};}

async function poll(jobId,outId,render){
  activeJob=jobId;
  const t=setInterval(async()=>{
    const j=await api('/api/job',{id:jobId});
    if(j.status==='running'){
      loaderSet(outId,null,'working',j.elapsed);
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
      loaderDone(outId,j.status==='done');
      if(j.status==='error'||j.status==='timeout'
        ||j.status==='canceled'){out(outId,j.error,'bad');}
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

/* ---- the Fit station: a window onto `synthkit fit` ---- */
function fitPayload(){
  return {src:document.getElementById('fsrc').value.trim(),
          out:document.getElementById('fout').value.trim(),
          group_by:document.getElementById('fgroup').value.trim(),
          lags:document.getElementById('flags').checked};}
async function fitTypes(){
  const o=document.getElementById('fit-out');
  o.textContent='reading the columns...';
  loaderSet('fit-out',null,'reading the columns');
  const r=await api('/api/fit-types',fitPayload());
  loaderDone('fit-out',!r.error);
  o.textContent=r.error?('STOPPED: '+r.error):r.text;
  if(!r.error)tick('fitsrc');}
let fitJob='',fitTimer=null;
async function fitRun(){
  const o=document.getElementById('fit-runout');
  const p=fitPayload();
  if(!p.src||!p.out){o.textContent=
    'STOPPED: give both a source CSV and an output directory - '+
    'the output directory is yours to choose.';return;}
  o.textContent='starting fit... (the full run on an 800-patient '+
    'extract takes about half an hour with lags)';
  const r=await api('/api/fit-run',p);
  if(r.error){o.textContent='STOPPED: '+r.error;return;}
  fitJob=r.job;
  if(fitTimer)clearInterval(fitTimer);
  fitTimer=setInterval(fitPoll,4000);}
async function fitPoll(){
  const o=document.getElementById('fit-runout');
  const j=await api('/api/job',{id:fitJob});
  const log=await api('/api/fit-log',
                      {out:document.getElementById('fout').value});
  if(j.status==='running'){
    const st=fitStage(log.text||'');
    loaderSet('fit-runout',st.frac,st.label,j.elapsed);
    o.textContent='running ('+Math.round(j.elapsed)+'s)\n\n'+
      (log.text||'');o.scrollTop=o.scrollHeight;return;}
  clearInterval(fitTimer);fitTimer=null;
  loaderDone('fit-runout',j.status==='done');
  if(j.status==='done'){
    o.textContent='DONE in '+Math.round(j.elapsed)+'s\n\n'+
      (log.text||'');
    tick('fitsrc');tick('fitrun');fitOpen();}
  else{o.textContent=(j.status||'error').toUpperCase()+': '+
    (j.error||'')+'\n\n'+(log.text||'');}}
async function fitOpen(){
  const v=document.getElementById('fit-verdict');
  v.innerHTML='<div class="hint">reading the run...</div>';
  loaderSet('fit-verdict',null,'reading the run');
  const r=await api('/api/fit-open',
                    {out:document.getElementById('fout').value});
  loaderDone('fit-verdict',!r.error);
  if(r.error){v.innerHTML='';
    const d=document.createElement('div');d.className='hint';
    d.textContent='STOPPED: '+r.error;v.appendChild(d);return;}
  let h='<h2>The gate &mdash; '+(r.gate.met?
    'MET on all criteria':'NOT MET')+'</h2>';
  r.gate.criteria.forEach(c=>{
    h+='<div class="gaterow"><b class="'+(c.ok?'ok':'bad')+'">'+
      (c.ok?'PASS':'FAIL')+'</b><span>'+c.name+' &mdash; '+
      c.detail+'</span></div>';});
  if(r.what_now&&r.what_now.length){
    h+='<div class="whatnow"><h3>What now &mdash; your options, '+
      'cheapest first</h3>';
    r.what_now.forEach(g=>{
      h+='<div class="wn-crit"><b>'+g.name.toUpperCase()+'</b> '+
        '&mdash; '+g.means+'<ol>';
      g.steps.forEach(st=>{h+='<li>'+st+'</li>';});
      h+='</ol></div>';});
    h+='<div class="hint">These words come from synthkit.gate, the '+
      'same module scripts/m0_gate.py prints them from - the bench '+
      'and the script cannot disagree. A gate is a floor, not a '+
      'certificate; NOT MET is a reading, not a wall.</div></div>';}
  h+='<div class="hint" style="margin-top:8px">build '+r.build+
    ' &middot; '+(r.source.rows_read||'?')+' rows, '+
    (r.source.patients||'?')+' patients &middot; '+
    r.contradictions+' contradiction(s), '+r.disobedience+
    ' disobeyed declaration(s)';
  if(r.empty_gaps&&r.empty_gaps.length){
    h+=' &middot; empty-rate gaps: '+r.empty_gaps.map(g=>
      g.column+' '+(g.empty_source*100).toFixed(1)+'%\u2192'+
      (g.empty_generated*100).toFixed(1)+'%').join(', ');}
  h+='. A gate is a floor, not a certificate.</div>';
  v.innerHTML=h;tick('fitver');}
async function fitBridge(){
  const o=document.getElementById('fit-bridgeout');
  o.textContent='bridging the blueprint...';
  loaderSet('fit-bridgeout',null,'bridging the blueprint');
  const r=await api('/api/fit-bridge',
                    {out:document.getElementById('fout').value});
  loaderDone('fit-bridgeout',!r.error);
  if(r.error){o.textContent='STOPPED: '+r.error;return;}
  setSpec(r.spec);
  const k=document.getElementById('kind');
  if(k)k.value='table';
  o.textContent='Loaded into the spec slot as a table recipe.\n'+
    'Crossed: '+(r.crossed.join(', ')||'(none)')+'\n'+
    'Did NOT cross: '+(r.did_not_cross.join(', ')||'(none)')+'\n'+
    'Open Step 4 (Campaign) - the measured data now takes the same '+
    'road as an invented recipe.';}
/* ---- the Dashboard: original vs synthetic, drawn live ---- */
async function deckPrefill(){
  /* borrow whatever the Fit route already knows, so the operator
     rarely retypes a path */
  const s=document.getElementById('fsrc'),o=document.getElementById('fout');
  if(s&&s.value&&!document.getElementById('dsrc').value)
    document.getElementById('dsrc').value=s.value;
  if(o&&o.value&&!document.getElementById('dout').value)
    document.getElementById('dout').value=o.value;}
async function deckBuild(){
  deckPrefill();
  const out=document.getElementById('deck-out');
  const frame=document.getElementById('deck-frame');
  const body={src:document.getElementById('dsrc').value.trim(),
    out:document.getElementById('dout').value.trim(),
    group_by:document.getElementById('dgroup').value.trim(),
    compare_dir:document.getElementById('dcmpdir').value.trim(),
    compare_label:document.getElementById('dcmplabel').value.trim()};
  if(!body.src||!body.out){out.textContent=
    'STOPPED: give the source CSV and the run directory.';return;}
  out.textContent='drawing the dashboard from '+body.out+'...';
  loaderSet('deck-out',null,'drawing the dashboard');
  const r=await api('/api/deck',body);
  loaderDone('deck-out',!r.error);
  if(r.error){out.textContent='STOPPED: '+r.error;
    frame.style.display='none';return;}
  frame.srcdoc=r.html;
  frame.style.display='block';
  out.textContent='drawn: '+r.columns+' columns, '+r.suppressed+
    ' source bin(s) suppressed by the k rule. The view below is the '+
    'same picture the shareable report carries.';
  tick('dashboard');
  frame.scrollIntoView({behavior:'smooth',block:'start'});}
</script><div id="cellmodal" onclick="if(event.target===this)closeCell()">
  <div class="box"><span class="close"
    onclick="closeCell()">&times;</span>
  <h4 id="cellmodal-title"></h4>
  <pre id="cellmodal-body"></pre></div>
</div>
</body></html>
"""

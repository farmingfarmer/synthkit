# Plugging a real vendor's model into the showdown

The showdown judges any model that can produce one risk score
per test record. The vendor never sees the answer key; the exam
is identical for every contestant. Three integration routes, in
increasing formality:

## Route 1 — wrapper file (most common)
The vendor (or you) writes a small Python file exposing one
function with the campaign contract and places it next to
synthkit (the repo root). Type `filename:function` at Step 5.2.

    # acme_readmit.py  ->  vendor box: acme_readmit:predict
    from acme_sdk import RiskModel        # vendor's library

    def predict(train_rows, train_labels, test_rows):
        model = RiskModel().fit(train_rows, train_labels)
        return [model.score(r) for r in test_rows]

`train_rows`/`test_rows` are lists of dicts (column -> string,
messy exactly as rendered); `train_labels` is a list of bools;
return a list of floats (higher = higher risk), one per test
row, in order.

## Route 2 — API wrapper (nothing installed)
Same file shape; the function calls the vendor's hosted scoring
endpoint instead of local code:

    import json, urllib.request

    def predict(train_rows, train_labels, test_rows):
        req = urllib.request.Request(
            "https://scoring.vendor.com/v1/score",
            data=json.dumps({"rows": test_rows}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer <key>"})
        with urllib.request.urlopen(req) as r:
            return json.load(r)["scores"]

(If the vendor's product does not train on customer data, the
train arguments are simply ignored — that is itself a fact
worth noting in the evaluation.)

## Route 3 — offline scoring exchange (no integration at all)
For vendors who will not run code or accept calls:

1. Render the population at Step 3 and download the synthetic
   data CSV. Delete the outcome column before sending.
2. The vendor returns one score per row, same order, as a CSV
   with a single `score` column (or one number per line).
3. Wrap the returned file:

       # vendor_scores.py -> vendor box: vendor_scores:predict
       def predict(train_rows, train_labels, test_rows):
           with open("acme_scores.csv", encoding="utf-8") as f:
               lines = [l.strip() for l in f if l.strip()]
           if lines and not lines[0].replace(".", "", 1)\
                   .replace("-", "", 1).isdigit():
               lines = lines[1:]              # header
           return [float(v) for v in lines]

Caveat for Route 3: the showdown exams three difficulty tiers,
each with its own test population — a static file can only
answer one. Use it for a single-tier check, or repeat the
exchange per tier. Routes 1–2 handle all tiers automatically.

## What the vendor can and cannot see
- CAN: the messy training rows and training answers; the messy
  test rows.
- CANNOT: the clean answer key, the planted-truth ledger, the
  test answers, or the other contestant's scores.

That asymmetry is the entire point: an exam only works if the
answer key stays home.

"""The 'vendor' for the demo: VendorCo RiskScore(tm).

A competent, honestly-built tabular model — synthkit's own
encoder + logistic baseline — which is exactly what makes the
demonstration fair: the vendor is not a strawman. It loses
because it cannot read the discharge notes, not because it is
badly made. Claimed performance: AUROC >= 0.75.

    showdown vendor field:  vendor_model:predict
"""
from synthkit.autosolver import autosolver

_solver = autosolver()


def predict(train_rows, train_labels, test_rows):
    return _solver(train_rows, train_labels, test_rows)

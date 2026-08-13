"""Evaluation metrics for the triage model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def recall_at_fpr(y_true, y_score, target_fpr: float = 0.20) -> float:
    """Recall (true positive rate) at a given false positive rate on the
    ROC curve, linearly interpolated between the two curve points
    bracketing target_fpr (roc_curve's fpr is monotonically non-decreasing,
    so interpolation is well-defined).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))

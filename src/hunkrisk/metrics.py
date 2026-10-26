"""Agreement and quality metrics. `high` is the positive class throughout."""

from collections.abc import Sequence

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from hunkrisk.schema import CATEGORIES


def risk_metrics(reference: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    """Precision, recall and F1 on `high`, raw agreement, and Cohen's kappa."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        reference, predicted, pos_label="high", average="binary", zero_division=0
    )
    agreement = sum(r == p for r, p in zip(reference, predicted, strict=True)) / len(reference)
    kappa = (
        cohen_kappa_score(reference, predicted) if len(set(reference) | set(predicted)) > 1 else 1.0
    )
    return {
        "n": len(reference),
        "agreement": agreement,
        "kappa": float(kappa),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def category_f1(
    reference: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]
) -> dict[str, float | None]:
    """F1 per category, each treated as its own binary task. None when it never occurs."""
    scores: dict[str, float | None] = {}
    for category in CATEGORIES:
        ref = [category in labels for labels in reference]
        pred = [category in labels for labels in predicted]
        if not any(ref) and not any(pred):
            scores[category] = None
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            ref, pred, pos_label=True, average="binary", zero_division=0
        )
        scores[category] = float(f1)
    return scores

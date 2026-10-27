"""
graph_src.evaluation — Metrics + evaluation runners.

Self-contained in graph_src — no cross-package imports.
"""

from graph_src.evaluation.base_metrics import (
    compute_all_metrics,
    exact_match,
    normalize_answer,
    token_f1,
)
from graph_src.evaluation.domain_metrics import (
    evaluate_domain,
    DOMAIN_EVAL_FN,
    DOMAIN_PRIMARY_METRIC_NAME,
)

__all__ = [
    "compute_all_metrics",
    "exact_match",
    "normalize_answer",
    "token_f1",
    "evaluate_domain",
    "DOMAIN_EVAL_FN",
    "DOMAIN_PRIMARY_METRIC_NAME",
]

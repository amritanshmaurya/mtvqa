from graph_src.evaluation.base_metrics import (
    compute_all_metrics,
    compute_bertscore,
    compute_retrieval_recall,
    exact_match,
    normalize_answer,
    token_f1,
)

from graph_src.evaluation.domain_metrics import (
    evaluate_domain,
    DOMAIN_EVAL_FN,
    DOMAIN_PRIMARY_METRIC_NAME,
    wtq_denotation_accuracy,
    vtabfact_accuracy,
    fintabnetqa_accuracy,
)

from graph_src.evaluation.answer_normalizer import (
    normalize_for_match,
    normalized_exact_match,
    normalized_substring_match,
    compute_normalized_em,
    compute_normalized_substring_acc,
)

def graph_utilization_stats(records: list) -> dict:

    total = len(records)
    if total == 0:
        return {"total": 0}

    method_counts = {}
    for r in records:
        method = r.get("method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    stats = {
        "total": total,
        "method_distribution": {
            m: {"count": c, "pct": round(100 * c / total, 1)}
            for m, c in sorted(method_counts.items())
        },
    }


    multi_methods = {"multi_patch_vlm", "multi_node_pot"}
    multi_count = sum(
        method_counts.get(m, 0) for m in multi_methods
    )
    stats["multi_table_usage_rate"] = round(multi_count / total, 4)


    graph_methods = {"multi_patch_vlm", "multi_node_pot", "single_pot"}
    graph_count = sum(
        method_counts.get(m, 0) for m in graph_methods
    )
    stats["graph_pipeline_rate"] = round(graph_count / total, 4)

    return stats


__all__ = [
    "compute_all_metrics",
    "compute_bertscore",
    "compute_retrieval_recall",
    "exact_match",
    "normalize_answer",
    "token_f1",
    "evaluate_domain",
    "DOMAIN_EVAL_FN",
    "DOMAIN_PRIMARY_METRIC_NAME",
    "wtq_denotation_accuracy",
    "vtabfact_accuracy",
    "fintabnetqa_accuracy",
    "graph_utilization_stats",
    "normalize_for_match",
    "normalized_exact_match",
    "normalized_substring_match",
    "compute_normalized_em",
    "compute_normalized_substring_acc",
]

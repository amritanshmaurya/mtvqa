import argparse
import gc
import torch
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_generator_class(model_path: str):

    lp = model_path.lower()
    if "molmo" in lp:
        from graph_src.generation.molmo_qa import TableQAGenerator
    elif "pixtral" in lp:
        from graph_src.generation.pixtral_qa import TableQAGenerator
    elif "phi" in lp and "vision" in lp:
        from graph_src.generation.phi_qa import TableQAGenerator
    elif "internvl" in lp:
        from graph_src.generation.internvl_qa import TableQAGenerator
    else:
        from graph_src.generation.qwen_qa import TableQAGenerator
    return TableQAGenerator
from graph_src.indexing.table_structural_index import TableStructuralIndexBuilder
from graph_src.routing.structural_router import StructuralRouter

from graph_src.evaluation.metrics import (
    compute_all_metrics,
    evaluate_domain,
    DOMAIN_PRIMARY_METRIC_NAME,
    graph_utilization_stats,
)
from graph_src.graph.graph_store import GraphStore
from graph_src.indexing.graph_indexer import GraphIndexer
from graph_src.retrieval.graph_retriever import GraphRetriever
from graph_src.routing.graph_router import GraphRouter
from graph_src.routing.crop_and_answer import GraphCropAndAnswer


def load_tablevqa_dataset(
    dataset_dir: str,
    domains: Optional[List[str]] = None,
) -> List[Dict]:

    json_path = os.path.join(dataset_dir, "tablevqa_bench.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset not found at {json_path}. "
            "Expected tablevqa_bench.json in the dataset directory."
        )
    with open(json_path, "r") as f:
        samples = json.load(f)
    if domains:
        samples = [s for s in samples if s.get("domain") in domains]
    print(f"Loaded {len(samples)} samples from TableVQA-Bench")
    if not domains:
        domain_counts = {}
        for s in samples:
            d = s.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c} samples")
    return samples


def normalize_vtabfact_output(answer: str) -> str:

    a = answer.strip().lower()
    if a in ("true", "1", "yes", "correct", "supported", "entailed"):
        return "1"
    if a in ("false", "0", "no", "incorrect", "refuted", "not supported",
             "not entailed", "wrong"):
        return "0"
    if a.startswith("true") or a.startswith("yes"):
        return "1"
    if a.startswith("false") or a.startswith("no"):
        return "0"
    if "true" in a and "false" not in a:
        return "1"
    if "false" in a and "true" not in a:
        return "0"
    return answer


def evaluate_per_domain(
    samples: List[Dict],
    predictions: List[str],
    references: List[str],
    device: str = "cuda:0",
) -> Dict[str, Dict]:

    domain_indices = defaultdict(list)
    for i, sample in enumerate(samples):
        domain_indices[sample.get("domain", "unknown")].append(i)

    results = {}
    for domain, indices in sorted(domain_indices.items()):
        dpreds = [predictions[i] for i in indices]
        drefs = [references[i] for i in indices]

        domain_result = evaluate_domain(domain, dpreds, drefs)
        supplementary = compute_all_metrics(
            dpreds, drefs, compute_bert=True, device=device,
        )
        merged = {**supplementary, **domain_result}
        merged["n_samples"] = len(indices)
        merged["metric_name"] = DOMAIN_PRIMARY_METRIC_NAME.get(
            domain, "Exact Match"
        )
        results[domain] = merged
    return results


def run_graph_pipeline_tablevqa(
    samples: List[Dict],
    all_maps: Dict,
    vlm,
    graph_store: GraphStore,
    images_dir: str,
    maps_dir: str,
    max_samples: int = None,
    max_tables: int = 2,
) -> Dict:

    router = GraphRouter()
    retriever = GraphRetriever(
        graph_store=graph_store,
        max_tables=max_tables,
        k_hops=2,
    )
    answerer = GraphCropAndAnswer(vlm_generator=vlm)
    structural_router = StructuralRouter()

    predictions, references, records = [], [], []
    samples_to_eval = samples[:max_samples] if max_samples else samples

    stats = {
        "total": 0, "graph_route": 0,
        "structural_fallback": 0, "map_missing": 0,
        "question_types": {},
    }

    for sample in tqdm(samples_to_eval, desc="Graph Pipeline (TableVQA)"):
        stats["total"] += 1
        domain = sample.get("domain", "")
        question = sample["question"]
        answer_gt = sample["answer"]
        table_id = sample.get("table_id", "")
        image_path = sample.get("image_path", "")
        stem = (
            Path(table_id).stem if table_id
            else Path(image_path).stem if image_path else ""
        )


        analysis = router.classify(question)
        qtype = str(analysis["question_type"])
        stats["question_types"][qtype] = (
            stats["question_types"].get(qtype, 0) + 1
        )


        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            predictions.append("")
            references.append(answer_gt)
            records.append({
                "table_id": table_id, "error": str(e), "pred": "",
                "method": "error", "domain": domain,
            })
            continue

        smap = all_maps.get(stem)


        if smap is not None and graph_store.num_nodes > 0:
            ranked_tables = retriever.retrieve(
                question=question,
                analysis=analysis,
                image_stem=stem,
            )

            if ranked_tables:
                result = answerer.answer(
                    question=question,
                    ranked_tables=ranked_tables,
                    maps_dir=maps_dir,
                    images_dir=images_dir,
                    analysis=analysis,
                    domain=domain,
                )
                pred = result.get("answer", "")
                method = result.get("method", "unknown")
                stats["graph_route"] += 1


                if not pred or method in ("no_images", "oom_skip", "no_tables"):
                    try:
                        pred = answerer.answer_single(
                            question=question, full_image=img,
                            crop_bbox=[0, 0, img.width, img.height],
                            domain=domain,
                        )
                        method = "full_image_fallback"
                    except Exception:
                        pred = ""
                    if not pred:
                        method = "empty_fallback"
                    stats["structural_fallback"] += 1
            else:

                try:
                    pred = answerer.answer_single(
                        question=question, full_image=img,
                        crop_bbox=[0, 0, img.width, img.height],
                        domain=domain,
                    )
                    method = "full_image_fallback"
                except Exception:
                    pred = ""
                    method = "empty_fallback"
                stats["structural_fallback"] += 1
        elif smap is not None:
            routing = structural_router.route(question, smap, img)
            pred = answerer.answer_single(
                question=question, full_image=img,
                crop_bbox=routing["region_bbox"], domain=domain,
            )
            method = "structural_only"
            stats["structural_fallback"] += 1
        else:
            stats["map_missing"] += 1
            stats["structural_fallback"] += 1
            pred = answerer.answer_single(
                question=question, full_image=img,
                crop_bbox=[0, 0, img.width, img.height], domain=domain,
            )
            method = "no_map_fallback"

        if domain == "vtabfact":
            pred = normalize_vtabfact_output(pred)

        predictions.append(pred)
        references.append(answer_gt)
        records.append({
            "table_id": table_id, "question": question,
            "gt": answer_gt, "pred": pred,
            "method": method, "domain": domain,
            "question_type": qtype,
        })

    return {
        "predictions": predictions,
        "references": references,
        "records": records,
        "stats": stats,
    }


def _delta_str(a, b):
    d = a - b
    sign = "▲+" if d >= 0 else "▼"
    return f"{sign}{abs(d):.4f}"


def _weighted_primary(per_domain_dict):
    total_n, total_correct = 0, 0.0
    for domain, dm in per_domain_dict.items():
        n = dm.get("n_samples", 0)
        pa = dm.get("primary_accuracy", dm.get("exact_match", 0))
        total_n += n
        total_correct += pa * n
    return total_correct / max(1, total_n)


def main():
    parser = argparse.ArgumentParser(
        description="Graph pipeline eval on TableVQA-Bench"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--maps_dir", required=True)
    parser.add_argument(
        "--graph_path",
        default="./data/indices/table_graph_tablevqa.gpickle",
    )
    parser.add_argument("--output_dir", default="./results/graph_tablevqa")
    parser.add_argument("--domains", default=None,
                        help="Comma-separated domains (default: all 4)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--build_graph", action="store_true")
    parser.add_argument("--max_tables", type=int, default=2)
    parser.add_argument("--min_edge_weight", type=float, default=0.08)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    domains_list = args.domains.split(",") if args.domains else None


    samples = load_tablevqa_dataset(args.dataset_dir, domains=domains_list)
    images_dir = os.path.join(args.dataset_dir, "images")
    valid = [
        s for s in samples
        if s.get("image_path") and os.path.exists(s["image_path"])
    ]
    if len(valid) < len(samples):
        print(f"Warning: {len(samples) - len(valid)} missing images "
              f"({len(valid)} usable)")
    samples = valid
    if not samples:
        print("ERROR: No valid samples.")
        return


    all_maps = TableStructuralIndexBuilder.load_all_maps(args.maps_dir)
    print(f"Loaded {len(all_maps)} structural maps")


    if args.build_graph or not os.path.exists(args.graph_path):
        print("Building table graph...")
        indexer = GraphIndexer(
            use_embeddings=False,
            min_edge_weight=args.min_edge_weight,
        )
        graph_store = indexer.build_from_maps(
            maps_dir=args.maps_dir,
            output_path=args.graph_path,
        )
    else:
        graph_store = GraphStore.load(args.graph_path)


    GeneratorClass = _load_generator_class(args.model_path)
    vlm = GeneratorClass(
        model_path=args.model_path,
        device=args.device,
        load_in_4bit=False,
    )


    graph_out = run_graph_pipeline_tablevqa(
        samples=samples,
        all_maps=all_maps,
        vlm=vlm,
        graph_store=graph_store,
        images_dir=images_dir,
        maps_dir=args.maps_dir,
        max_samples=args.max_samples,
        max_tables=args.max_tables,
    )


    eval_samples = samples[:args.max_samples] if args.max_samples else samples
    graph_per_domain = evaluate_per_domain(
        eval_samples,
        graph_out["predictions"],
        graph_out["references"],
        device=args.device,
    )
    graph_primary_overall = _weighted_primary(graph_per_domain)
    util_stats = graph_utilization_stats(graph_out["records"])
    stats = graph_out["stats"]
    total = max(1, stats["total"])


    print("\n" + "=" * 80)
    print(f"  TABLEVQA-BENCH — Graph Pipeline — "
          f"{args.model_path.split('/')[-1]}")
    print("=" * 80)

    all_domains = sorted(set(s.get("domain", "unknown") for s in eval_samples))
    print(f"{'Domain':<16} {'Metric':<17} {'Score':>10} {'N':>5}")
    print("─" * 55)
    for domain in all_domains:
        dm = graph_per_domain.get(domain, {})
        metric_name = dm.get("metric_name", "Exact Match")
        pa = dm.get("primary_accuracy", dm.get("exact_match", 0))
        n = dm.get("n_samples", 0)
        print(f"{domain:<16} {metric_name:<17} {pa * 100:>9.1f}% {n:>5}")
        print(f"{'':>16} {'  (EM)':<17} "
              f"{dm.get('exact_match', 0) * 100:>9.1f}%")
    print("─" * 55)
    print(f"{'OVERALL':<16} {'Weighted Primary':<17} "
          f"{graph_primary_overall * 100:>9.1f}%")

    print(f"\nRouting: {stats['graph_route']} graph / "
          f"{stats['structural_fallback']} fallback / "
          f"{stats['map_missing']} map_missing")

    print(f"\nQuestion types:")
    for qtype, count in sorted(stats["question_types"].items()):
        print(f"  {qtype:<20} {count} ({count / total * 100:.1f}%)")

    print(f"\nGraph utilization:")
    print(f"  Multi-table: {util_stats.get('multi_table_usage_rate', 0) * 100:.1f}%")
    print(f"  Graph pipeline: {util_stats.get('graph_pipeline_rate', 0) * 100:.1f}%")


    print("=" * 80)

    # Save JSON
    save_data = {
        "model": args.model_path,
        "pipeline": "graph",
        "evaluation_type": "domain_specific",
        "graph_stats": graph_store.get_stats(),
        "graph_metrics": {
            "weighted_primary_overall": graph_primary_overall,
            "per_domain": graph_per_domain,
        },
        "routing_stats": stats,
        "utilization_stats": util_stats,
        "beats_target": beats,
        "records": graph_out["records"],
    }
    out_path = os.path.join(args.output_dir, "tablevqa_graph_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

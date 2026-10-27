import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

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
    graph_utilization_stats,
    compute_normalized_em,
    compute_normalized_substring_acc,
)
from graph_src.graph.graph_store import GraphStore
from graph_src.indexing.graph_indexer import GraphIndexer
from graph_src.retrieval.graph_retriever import GraphRetriever
from graph_src.routing.graph_router import GraphRouter
from graph_src.routing.crop_and_answer import GraphCropAndAnswer


def load_comtqa_dataset(dataset_dir: str) -> List[dict]:
    """Load ComTQA dataset from local JSON or HuggingFace."""
    images_dir = os.path.join(dataset_dir, "images")
    json_path  = os.path.join(dataset_dir, "comtqa.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = []
        for item in raw:
            image_name = item.get("image_name", "")
            table_id   = item.get("table_id", "")
            if not image_name and table_id:
                image_name = f"{table_id}.png"
            image_path = (
                os.path.join(images_dir, image_name) if image_name else ""
            )
            samples.append({
                "image_path": image_path,
                "question":   item["question"],
                "answer":     item["answer"],
                "table_id":   image_name,
                "dataset":    item.get("dataset", ""),
            })
        print(f"Loaded {len(samples)} QA pairs from {json_path}")
        return samples

    try:
        from datasets import load_dataset
        ds = load_dataset("ByteDance/ComTQA", split="train")
        samples = []
        for item in ds:
            image_name = item.get("image_name", "")
            table_id   = item.get("table_id", "")
            if not image_name and table_id:
                image_name = f"{table_id}.png"
            image_path = (
                os.path.join(images_dir, image_name) if image_name else ""
            )
            samples.append({
                "image_path": image_path,
                "question":   item["question"],
                "answer":     item["answer"],
                "table_id":   image_name,
                "dataset":    item.get("dataset", ""),
            })
        print(f"Loaded {len(samples)} QA pairs from HuggingFace")
        return samples
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load ComTQA from {json_path} or HuggingFace: {e}"
        )


def _substring_acc(predictions, references):
    """Normalized substring accuracy (bidirectional, with full normalization)."""
    return compute_normalized_substring_acc(predictions, references)


def run_graph_pipeline(
    samples: List[dict],
    all_maps: Dict[str, dict],
    vlm,
    graph_store: GraphStore,
    images_dir: str,
    maps_dir: str,
    max_samples: int = None,
    max_tables: int = 2,
) -> dict:

    router           = GraphRouter()
    retriever        = GraphRetriever(
        graph_store=graph_store,
        max_tables=max_tables,
        k_hops=2,
    )
    answerer         = GraphCropAndAnswer(vlm_generator=vlm)
    structural_router = StructuralRouter()

    predictions, references, records = [], [], []
    samples_to_eval = samples[:max_samples] if max_samples else samples

    stats = {
        "total":          0,
        "graph_route":    0,
        "structural_fallback": 0,
        "map_missing":    0,
        "question_types": {},
    }

    for sample in tqdm(samples_to_eval, desc="Graph Pipeline"):
        stats["total"] += 1
        question  = sample["question"]
        answer_gt = sample["answer"]
        table_id  = sample.get("table_id", "")
        image_path = sample.get("image_path", "")
        stem      = Path(table_id).stem if table_id else ""

        # Step 1: Classify question
        analysis = router.classify(question)
        qtype    = str(analysis["question_type"])
        stats["question_types"][qtype] = (
            stats["question_types"].get(qtype, 0) + 1
        )


        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            predictions.append("")
            references.append(answer_gt)
            records.append({
                "table_id": table_id,
                "error":    str(e),
                "pred":     "",
                "method":   "error",
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
                )
                pred   = result.get("answer", "")
                method = result.get("method", "unknown")
                stats["graph_route"] += 1
            else:

                routing = structural_router.route(question, smap, img)
                pred    = answerer.answer_single(
                    question=question,
                    full_image=img,
                    crop_bbox=routing["region_bbox"],
                )
                method = "structural_fallback"
                stats["structural_fallback"] += 1

        elif smap is not None:

            routing = structural_router.route(question, smap, img)
            pred    = answerer.answer_single(
                question=question,
                full_image=img,
                crop_bbox=routing["region_bbox"],
            )
            method = "structural_only"
            stats["structural_fallback"] += 1
        else:

            stats["map_missing"]         += 1
            stats["structural_fallback"] += 1
            pred   = answerer.answer_single(
                question=question,
                full_image=img,
                crop_bbox=[0, 0, img.width, img.height],
            )
            method = "no_map_fallback"

        predictions.append(pred)
        references.append(answer_gt)
        records.append({
            "table_id":     table_id,
            "question":     question,
            "gt":           answer_gt,
            "pred":         pred,
            "method":       method,
            "question_type": qtype,
        })

    return {
        "predictions": predictions,
        "references":  references,
        "records":     records,
        "stats":       stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ComTQA evaluation with graph-enhanced pipeline"
    )
    parser.add_argument("--dataset_dir",  required=True)
    parser.add_argument("--maps_dir",     required=True)
    parser.add_argument(
        "--graph_path",
        default="./data/indices/table_graph_comtqa.gpickle",
    )
    parser.add_argument("--output_dir",  default="./results/graph_comtqa")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument(
        "--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--build_graph", action="store_true",
        help="Build table graph before evaluation",
    )
    parser.add_argument(
        "--max_tables", type=int, default=2,
        help="Max tables to retrieve per question",
    )
    parser.add_argument(
        "--min_edge_weight", type=float, default=0.08,
        help="Min edge weight for graph construction",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    samples    = load_comtqa_dataset(args.dataset_dir)
    images_dir = os.path.join(args.dataset_dir, "images")
    valid = [
        s for s in samples
        if s.get("image_path") and os.path.exists(s["image_path"])
    ]
    if len(valid) < len(samples):
        print(
            f"Warning: {len(samples) - len(valid)} missing images "
            f"({len(valid)} usable)"
        )
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


    graph_out = run_graph_pipeline(
        samples=samples,
        all_maps=all_maps,
        vlm=vlm,
        graph_store=graph_store,
        images_dir=images_dir,
        maps_dir=args.maps_dir,
        max_samples=args.max_samples,
        max_tables=args.max_tables,
    )


    graph_metrics = compute_all_metrics(
        graph_out["predictions"],
        graph_out["references"],
        compute_bert=True,
        device=args.device,
    )

    norm_em = compute_normalized_em(
        graph_out["predictions"], graph_out["references"]
    )
    norm_subacc = compute_normalized_substring_acc(
        graph_out["predictions"], graph_out["references"]
    )
    graph_metrics["exact_match"] = norm_em
    graph_metrics["substring_acc"] = norm_subacc
    graph_subacc = norm_subacc


    util_stats = graph_utilization_stats(graph_out["records"])


    stats = graph_out["stats"]
    total = max(1, stats["total"])


    print("\n" + "=" * 60)
    print(f"  COMTQA — Graph Pipeline — {args.model_path.split('/')[-1]}")
    print("=" * 60)

    for metric in ["substring_acc", "exact_match", "token_f1", "bertscore_f1"]:
        val = graph_metrics.get(metric, 0)
        if metric in ("substring_acc", "exact_match"):
            print(f"  {metric:<20} {val * 100:.1f}%")
        else:
            print(f"  {metric:<20} {val:.4f}")

    print(f"\nRouting breakdown:")
    print(
        f"  Graph route    : {stats['graph_route']} "
        f"({stats['graph_route'] / total * 100:.1f}%)"
    )
    print(
        f"  Struct fallback: {stats['structural_fallback']} "
        f"({stats['structural_fallback'] / total * 100:.1f}%)"
    )
    print(
        f"  Map missing    : {stats['map_missing']} "
        f"({stats['map_missing'] / total * 100:.1f}%)"
    )

    print(f"\nQuestion type distribution:")
    for qtype, count in sorted(stats["question_types"].items()):
        print(f"  {qtype:<20} {count} ({count / total * 100:.1f}%)")

    print(f"\nGraph utilization:")
    print(
        f"  Multi-table usage : "
        f"{util_stats.get('multi_table_usage_rate', 0) * 100:.1f}%"
    )
    print(
        f"  Graph pipeline    : "
        f"{util_stats.get('graph_pipeline_rate', 0) * 100:.1f}%"
    )

    print("=" * 60)


    save_data = {
        "model":            args.model_path,
        "pipeline":         "graph",
        "graph_stats":      graph_store.get_stats(),
        "metrics":          graph_metrics,
        "routing_stats":    stats,
        "utilization_stats": util_stats,
        "beats_target":     beats,
        "records":          graph_out["records"],
    }
    out_path = os.path.join(args.output_dir, "comtqa_graph_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

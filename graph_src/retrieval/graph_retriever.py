import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_src.graph.entity_linker import extract_entities, _normalize_terms
from graph_src.graph.graph_store import GraphStore


class GraphRetriever:

    def __init__(
        self,
        graph_store: GraphStore,
        max_tables: int = 3,
        min_relevance: float = 0.05,
        k_hops: int = 2,
    ):
        self.store = graph_store
        self.max_tables = max_tables
        self.min_relevance = min_relevance
        self.k_hops = k_hops

    def retrieve(
        self,
        question: str,
        analysis: Dict[str, Any],
        image_stem: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        needs_multi = analysis.get("needs_multi_table", False)
        is_temporal = analysis.get("comparison_type") == "temporal"


        q_entities = self._extract_question_entities(question, analysis)

        if is_temporal and analysis.get("years_mentioned"):
            results = self._temporal_retrieve(
                analysis, q_entities, image_stem
            )
            if results:
                return results[:self.max_tables]

        if needs_multi:
            results = self._multi_table_retrieve(
                q_entities, analysis, image_stem
            )
        else:
            results = self._single_table_retrieve(
                q_entities, analysis, image_stem
            )

        return results[:self.max_tables]


    def _single_table_retrieve(
        self,
        q_entities: Set[str],
        analysis: Dict[str, Any],
        image_stem: Optional[str],
    ) -> List[Dict[str, Any]]:

        if image_stem and image_stem in self.store.graph:
            return [{
                "image_stem": image_stem,
                "score": 1.0,
                "reason": "direct_known",
                "shared_entities": [],
            }]


        scored = self._score_all_nodes(q_entities)
        if not scored:
            return []

        return [scored[0]]  


    def _multi_table_retrieve(
        self,
        q_entities: Set[str],
        analysis: Dict[str, Any],
        image_stem: Optional[str],
    ) -> List[Dict[str, Any]]:

        if image_stem and image_stem in self.store.graph:
            seed = image_stem
            seed_score = 1.0
        else:
            scored = self._score_all_nodes(q_entities)
            if not scored:
                return []
            seed = scored[0]["image_stem"]
            seed_score = scored[0]["score"]

        results = [{
            "image_stem": seed,
            "score": seed_score,
            "reason": "seed_node",
            "shared_entities": [],
        }]


        neighbors = self.store.k_hop_neighbors(
            seed, k=self.k_hops, min_edge_weight=self.min_relevance
        )

        neighbor_scores = []
        for nid in neighbors:
            node_data = self.store.get_node(nid)
            if not node_data:
                continue

            node_entities = node_data.get("entities", set())
            entity_score = self._entity_overlap_score(q_entities, node_entities)


            edge = self.store.get_edge_data(seed, nid)
            if not edge:
                edge = self.store.get_edge_data(nid, seed)
            edge_weight = edge.get("weight", 0) if edge else 0
            shared = edge.get("shared_entities", []) if edge else []


            combined = 0.6 * entity_score + 0.4 * edge_weight


            metrics = analysis.get("metrics_mentioned", [])
            node_headers = [h.lower() for h in node_data.get("headers", [])]
            node_rows = [r.lower() for r in node_data.get("row_labels", [])]
            metric_hits = sum(
                1 for m in metrics
                if any(m.lower() in h for h in node_headers + node_rows)
            )
            if metrics and metric_hits > 0:
                combined += 0.15 * (metric_hits / len(metrics))

            if combined >= self.min_relevance:
                neighbor_scores.append({
                    "image_stem": nid,
                    "score": round(combined, 4),
                    "reason": "graph_traversal",
                    "shared_entities": shared,
                })


        neighbor_scores.sort(key=lambda x: x["score"], reverse=True)
        results.extend(neighbor_scores[:self.max_tables - 1])

        return results


    def _temporal_retrieve(
        self,
        analysis: Dict[str, Any],
        q_entities: Set[str],
        image_stem: Optional[str],
    ) -> List[Dict[str, Any]]:

        years = analysis.get("years_mentioned", [])
        metrics = analysis.get("metrics_mentioned", [])

        if not years:
            return []

        results = []
        used_nodes = set()

        for year in years:
            best_node = None
            best_score = 0.0

            for node_id, data in self.store.graph.nodes(data=True):
                if node_id in used_nodes:
                    continue

                node_entities = data.get("entities", set())

                year_lower = str(year).lower()
                if not any(
                    year_lower in e.lower() or e.lower() in year_lower
                    for e in node_entities
                ):
                    continue

                node_headers = [
                    h.lower() for h in data.get("headers", [])
                ]
                node_rows = [
                    r.lower() for r in data.get("row_labels", [])
                ]
                all_labels = node_headers + node_rows

                metric_score = 0.0
                if metrics:
                    hits = sum(
                        1 for m in metrics
                        if any(m.lower() in lbl for lbl in all_labels)
                    )
                    metric_score = hits / len(metrics)

                entity_score = self._entity_overlap_score(q_entities, node_entities)

                score = 0.4 + 0.35 * metric_score + 0.25 * entity_score

                if image_stem and node_id == image_stem:
                    score += 0.1

                if score > best_score:
                    best_score = score
                    best_node = node_id

            if best_node:
                used_nodes.add(best_node)
                results.append({
                    "image_stem": best_node,
                    "score": round(best_score, 4),
                    "reason": f"temporal_year_{year}",
                    "shared_entities": [str(year)] + metrics,
                })

        return results


    def _score_all_nodes(
        self,
        q_entities: Set[str],
    ) -> List[Dict[str, Any]]:

        scored = []
        for node_id, data in self.store.graph.nodes(data=True):
            node_entities = data.get("entities", set())
            score = self._entity_overlap_score(q_entities, node_entities)
            if score >= self.min_relevance:
                scored.append({
                    "image_stem": node_id,
                    "score": round(score, 4),
                    "reason": "entity_match",
                    "shared_entities": sorted(
                        q_entities & {e.lower() for e in node_entities}
                    ),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    @staticmethod
    def _entity_overlap_score(
        q_entities: Set[str],
        node_entities: Set[str],
    ) -> float:

        if not q_entities or not node_entities:
            return 0.0

        q_lower = {e.lower() for e in q_entities}
        n_lower = {e.lower() for e in node_entities}
        shared = q_lower & n_lower

        if not q_lower:
            return 0.0
        recall = len(shared) / len(q_lower)

        union = q_lower | n_lower
        jaccard = len(shared) / len(union) if union else 0.0

        return 0.7 * recall + 0.3 * jaccard

    @staticmethod
    def _extract_question_entities(
        question: str,
        analysis: Dict[str, Any],
    ) -> Set[str]:

        entities: Set[str] = set()

        for year in analysis.get("years_mentioned", []):
            entities.add(str(year))
        for metric in analysis.get("metrics_mentioned", []):
            entities.add(metric.lower())
        entities.update(analysis.get("entities", set()))

        entities.update(_normalize_terms(question.split()))

        return entities

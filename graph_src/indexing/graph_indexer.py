import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_src.graph.entity_linker import (
    EntityLinker,
    extract_entities,
    extract_numeric_fingerprint,
)
from graph_src.graph.graph_store import GraphStore


class GraphIndexer:

    def __init__(
        self,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_edge_weight: float = 0.08,
        w_jaccard: float = 0.5,
        w_numeric: float = 0.3,
        w_embedding: float = 0.2,
    ):
        
        self.min_edge_weight = min_edge_weight
        self.linker = EntityLinker(
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            w_jaccard=w_jaccard,
            w_numeric=w_numeric,
            w_embedding=w_embedding,
        )

    def build_from_maps(
        self,
        maps_dir: str,
        output_path: Optional[str] = None,
    ) -> GraphStore:
        
        start = time.time()
        maps = self._load_maps(maps_dir)
        if not maps:
            print("[GraphIndexer] No structural maps found. Empty graph.")
            return GraphStore()

        print(f"[GraphIndexer] Building graph from {len(maps)} structural maps...")

        graph = nx.DiGraph()

        node_entities: Dict[str, set] = {}
        node_fingerprints: Dict[str, dict] = {}
        node_texts: Dict[str, str] = {}

        for stem, smap in maps.items():
            entities = extract_entities(smap)
            fingerprint = extract_numeric_fingerprint(smap)

            text_blob = " ".join([
                smap.get("title", ""),
                " ".join(smap.get("headers", [])),
                " ".join(smap.get("row_labels", [])),
            ]).strip()

            graph.add_node(
                stem,
                image_stem=stem,
                title=smap.get("title", "Unknown"),
                headers=smap.get("headers", []),
                row_labels=smap.get("row_labels", []),
                num_rows=smap.get("num_rows", 0),
                num_cols=smap.get("num_cols", 0),
                data_type=smap.get("data_type", "mixed"),
                data_summary=smap.get("data_summary", ""),
                entities=entities,
                numeric_fingerprint=fingerprint,
            )

            node_entities[stem] = entities
            node_fingerprints[stem] = fingerprint
            node_texts[stem] = text_blob

        stems = sorted(maps.keys())
        n = len(stems)
        edges_added = 0
        pairs_checked = 0

        for i in range(n):
            for j in range(i + 1, n):
                stem_a, stem_b = stems[i], stems[j]

                weight, shared, edge_type = self.linker.link(
                    entities_a=node_entities[stem_a],
                    entities_b=node_entities[stem_b],
                    fingerprint_a=node_fingerprints[stem_a],
                    fingerprint_b=node_fingerprints[stem_b],
                    text_a=node_texts[stem_a],
                    text_b=node_texts[stem_b],
                )
                pairs_checked += 1

                if weight >= self.min_edge_weight:
                    # Add bidirectional edges
                    edge_attrs = {
                        "weight": round(weight, 4),
                        "edge_type": edge_type,
                        "shared_entities": shared,
                    }
                    graph.add_edge(stem_a, stem_b, **edge_attrs)
                    graph.add_edge(stem_b, stem_a, **edge_attrs)
                    edges_added += 1

            if (i + 1) % 500 == 0:
                print(f"  [GraphIndexer] Processed {i + 1}/{n} nodes, "
                      f"{edges_added} edges so far...")

        elapsed = time.time() - start
        store = GraphStore(graph=graph)

        print(f"[GraphIndexer] Done in {elapsed:.1f}s")
        print(f"  Pairs checked: {pairs_checked}")
        print(f"  Edges added: {edges_added} (bidirectional)")
        print(f"  {store.summary()}")

        if output_path:
            store.save(output_path)

        return store

    def get_stats(self) -> dict:

        return {
            "use_embeddings": self.linker.use_embeddings,
            "min_edge_weight": self.min_edge_weight,
            "w_jaccard": self.linker.w_jaccard,
            "w_numeric": self.linker.w_numeric,
            "w_embedding": self.linker.w_embedding,
        }

    @staticmethod
    def _load_maps(maps_dir: str) -> Dict[str, dict]:
        maps = {}
        maps_path = Path(maps_dir)
        if not maps_path.is_dir():
            print(f"[GraphIndexer] WARNING: {maps_dir} is not a directory")
            return maps

        for p in sorted(maps_path.rglob("*.json")):
            # Skip log files and index files
            if p.name.endswith(".jsonl") or p.name == "index_log.jsonl":
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    maps[p.stem] = data
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [WARN] Skipping {p.name}: {e}")
                continue

        return maps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build table graph from structural maps."
    )
    parser.add_argument(
        "--maps_dir", required=True,
        help="Directory containing structural map JSON files",
    )
    parser.add_argument(
        "--output", default="data/indices/table_graph.gpickle",
        help="Output path for the graph file",
    )
    parser.add_argument(
        "--min_weight", type=float, default=0.08,
        help="Minimum edge weight threshold",
    )
    parser.add_argument(
        "--use_embeddings", action="store_true",
        help="Use sentence-transformer embeddings for linking",
    )
    args = parser.parse_args()

    indexer = GraphIndexer(
        use_embeddings=args.use_embeddings,
        min_edge_weight=args.min_weight,
    )
    store = indexer.build_from_maps(
        maps_dir=args.maps_dir,
        output_path=args.output,
    )
    print("\nFinal stats:")
    print(store.summary())

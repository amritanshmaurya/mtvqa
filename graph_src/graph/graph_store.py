import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx


class GraphStore:


    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph: nx.DiGraph = graph or nx.DiGraph()


    def save(self, output_path: str) -> None:

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[GraphStore] Saved graph to {output_path}")

    @classmethod
    def load(cls, graph_path: str) -> "GraphStore":

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        store = cls(graph=graph)
        print(f"[GraphStore] Loaded graph from {graph_path} "
              f"({store.num_nodes} nodes, {store.num_edges} edges)")
        return store

    def set_graph(self, graph: nx.DiGraph) -> None:

        self.graph = graph


    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()


    def get_node(self, node_id: str) -> Optional[dict]:

        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def k_hop_neighbors(
        self,
        node_id: str,
        k: int = 1,
        min_edge_weight: float = 0.0,
    ) -> List[str]:

        if node_id not in self.graph:
            return []

        visited: Set[str] = {node_id}
        frontier: Set[str] = {node_id}

        for _ in range(k):
            next_frontier: Set[str] = set()
            for current in frontier:

                for succ in self.graph.successors(current):
                    edge_data = self.graph.edges[current, succ]
                    if edge_data.get("weight", 0) >= min_edge_weight:
                        if succ not in visited:
                            next_frontier.add(succ)
                            visited.add(succ)

                for pred in self.graph.predecessors(current):
                    edge_data = self.graph.edges[pred, current]
                    if edge_data.get("weight", 0) >= min_edge_weight:
                        if pred not in visited:
                            next_frontier.add(pred)
                            visited.add(pred)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(node_id)
        return sorted(visited)

    def subgraph(self, node_ids: List[str]) -> nx.DiGraph:

        valid_ids = [n for n in node_ids if n in self.graph]
        return self.graph.subgraph(valid_ids).copy()

    def find_nodes_by_entity(
        self,
        entity: str,
        case_sensitive: bool = False,
    ) -> List[str]:

        target = entity if case_sensitive else entity.lower()
        matches = []
        for node_id, data in self.graph.nodes(data=True):
            node_entities = data.get("entities", set())
            if case_sensitive:
                if target in node_entities:
                    matches.append(node_id)
            else:
                if target in {e.lower() for e in node_entities}:
                    matches.append(node_id)
        return matches

    def get_edge_data(
        self,
        src: str,
        dst: str,
    ) -> Optional[dict]:

        if self.graph.has_edge(src, dst):
            return dict(self.graph.edges[src, dst])
        return None

    def get_edges_for_node(
        self,
        node_id: str,
        min_weight: float = 0.0,
    ) -> List[dict]:

        if node_id not in self.graph:
            return []

        edges = []
        # Outgoing
        for _, dst, data in self.graph.out_edges(node_id, data=True):
            if data.get("weight", 0) >= min_weight:
                edges.append({
                    "src": node_id,
                    "dst": dst,
                    **data,
                })
        # Incoming
        for src, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("weight", 0) >= min_weight:
                edges.append({
                    "src": src,
                    "dst": node_id,
                    **data,
                })
        return edges


    def get_stats(self) -> dict:

        if self.num_nodes == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "avg_degree": 0.0,
                "connected_components": 0,
                "edge_type_distribution": {},
            }

        degrees = dict(self.graph.degree())
        avg_degree = sum(degrees.values()) / self.num_nodes


        edge_types: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            etype = data.get("edge_type", "unknown")
            edge_types[etype] = edge_types.get(etype, 0) + 1

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "avg_degree": round(avg_degree, 2),
            "connected_components": nx.number_weakly_connected_components(
                self.graph
            ),
            "edge_type_distribution": edge_types,
        }

    def summary(self) -> str:

        stats = self.get_stats()
        lines = [
            f"TableGraph: {stats['num_nodes']} nodes, {stats['num_edges']} edges",
            f"  Avg degree: {stats['avg_degree']}",
            f"  Connected components: {stats['connected_components']}",
        ]
        for etype, count in stats.get("edge_type_distribution", {}).items():
            lines.append(f"  Edge type '{etype}': {count}")
        return "\n".join(lines)

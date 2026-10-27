"""
graph_src.indexing — Structural indexing + graph indexing.

Self-contained in graph_src — no cross-package imports.
"""


def __getattr__(name):
    """Lazy re-export to avoid torch import at module load time."""
    if name == "TableStructuralIndexBuilder":
        from graph_src.indexing.table_structural_index import (
            TableStructuralIndexBuilder,
        )
        return TableStructuralIndexBuilder
    raise AttributeError(f"module 'graph_src.indexing' has no attribute {name!r}")

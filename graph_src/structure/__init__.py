# graph_src.structure — self-contained
from graph_src.structure.table_detector import (
    TableStructureDetector,
    BBox,
    get_row_patches,
    get_header_patch,
    get_column_patches,
    get_full_table_patch,
    visualize_structure,
)

__all__ = [
    "TableStructureDetector",
    "BBox",
    "get_row_patches",
    "get_header_patch",
    "get_column_patches",
    "get_full_table_patch",
    "visualize_structure",
]

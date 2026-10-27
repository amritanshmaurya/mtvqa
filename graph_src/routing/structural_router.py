import re
from typing import Any, Dict, List, Tuple

from PIL import Image

_FULL_TABLE_PATTERNS = [
    r"\bhow many\b", r"\btotal\b", r"\bsum\b", r"\baverage\b",
    r"\bmean\b", r"\bcount\b", r"\ball rows\b", r"\beach row\b",
    r"\bevery row\b", r"\blist all\b", r"\benumerate\b",
    r"\brank\b", r"\boverall\b", r"\bacross all\b",
]
_SUPERLATIVE_PATTERNS = [
    r"\bhighest\b", r"\blowest\b", r"\blargest\b", r"\bsmallest\b",
    r"\bmaximum\b", r"\bminimum\b", r"\bmost\b", r"\bfewest\b",
    r"\bbest\b", r"\bworst\b", r"\bgreatest\b",
]


class StructuralRouter:

    def __init__(self):
        print("[StructuralRouter] Keyword routing — no model needed.")


    def route(
        self,
        question: str,
        structural_map: Dict[str, Any],
        image: Image.Image,
    ) -> Dict[str, Any]:

        w, h = image.size

        headers = structural_map.get("headers", [])
        row_labels = structural_map.get("row_labels", [])

        if not headers and not row_labels:
            return self._fallback(w, h)

        q = question.lower().strip()

        if self._needs_full_table(q):
            ft = structural_map.get("regions", {}).get(
                "full_table", {}
            ).get("bbox", [0, 0, w, h])
            return {
                "region_bbox": [int(x) for x in ft],
                "target_rows": ["ALL"],
                "target_cols": ["ALL"],
                "confidence": 0.8,
                "routing_method": "structural_full",
            }

        matched_rows = self._match_labels(q, row_labels)
        matched_cols = self._match_labels(q, headers)

        if not matched_rows and not matched_cols:
            return self._fallback(w, h)

        confidence = self._score_confidence(
            matched_rows, matched_cols, row_labels, headers
        )
        if confidence < 0.4:
            return self._fallback(w, h)

        bbox = self._bbox_from_routing(
            matched_rows, matched_cols, structural_map, (w, h)
        )

        return {
            "region_bbox": bbox,
            "target_rows": matched_rows,
            "target_cols": matched_cols,
            "confidence": confidence,
            "routing_method": "structural",
        }

    @staticmethod
    def _needs_full_table(q: str) -> bool:
        return any(
            re.search(p, q)
            for p in _FULL_TABLE_PATTERNS + _SUPERLATIVE_PATTERNS
        )

    @staticmethod
    def _match_labels(question: str, labels: List[str]) -> List[str]:

        if not labels:
            return []

        q_lower = question.lower()
        q_tokens = set(q_lower.split())
        matched = []
        seen = set()

        for label in labels:
            if label in seen:
                continue
            label_lower = label.lower().strip()
            if not label_lower:
                continue

            if label_lower in q_lower:
                matched.append(label)
                seen.add(label)
                continue

            label_tokens = [t for t in label_lower.split() if len(t) > 2]
            if label_tokens:
                overlap = sum(1 for t in label_tokens if t in q_tokens)
                if overlap >= max(1, len(label_tokens) * 0.5):
                    matched.append(label)
                    seen.add(label)
                    continue

        return matched

    @staticmethod
    def _score_confidence(
        matched_rows: List[str],
        matched_cols: List[str],
        all_rows: List[str],
        all_headers: List[str],
    ) -> float:
        if matched_rows and matched_cols:
            return max(
                0.5,
                0.9 - 0.3 * (len(matched_rows) / max(1, len(all_rows))),
            )
        if matched_rows or matched_cols:
            return 0.55
        return 0.0

    @staticmethod
    def _bbox_from_routing(
        matched_rows: List[str],
        matched_cols: List[str],
        structural_map: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> List[int]:

        w, h = image_size
        regions = structural_map.get("regions", {})

        row_bboxes: List[List[int]] = []
        for key, info in regions.items():
            if not key.startswith("row_"):
                continue
            label = info.get("label", key)
            for mr in matched_rows:
                if (mr.lower() == label.lower()
                        or mr.lower() in label.lower()
                        or label.lower() in mr.lower()):
                    row_bboxes.append(info["bbox"])
                    break

        col_bboxes: List[List[int]] = []
        for key, info in regions.items():
            if not key.startswith("col_"):
                continue
            label = info.get("label", key)
            for mc in matched_cols:
                if (mc.lower() == label.lower()
                        or mc.lower() in label.lower()
                        or label.lower() in mc.lower()):
                    col_bboxes.append(info["bbox"])
                    break

        if row_bboxes:
            x1 = min(b[0] for b in row_bboxes)
            y1 = min(b[1] for b in row_bboxes)
            x2 = max(b[2] for b in row_bboxes)
            y2 = max(b[3] for b in row_bboxes)
            if col_bboxes:
                x1 = min(b[0] for b in col_bboxes)
                x2 = max(b[2] for b in col_bboxes)
        elif col_bboxes:
            x1 = min(b[0] for b in col_bboxes)
            y1 = min(b[1] for b in col_bboxes)
            x2 = max(b[2] for b in col_bboxes)
            y2 = max(b[3] for b in col_bboxes)
        else:
            ft = regions.get("full_table", {}).get("bbox", [0, 0, w, h])
            return [int(x) for x in ft]

        hdr = regions.get("header_row", {}).get("bbox")
        if hdr:
            y1 = min(y1, hdr[1])
            x1 = min(x1, hdr[0])
            x2 = max(x2, hdr[2])

        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return [int(x1), int(y1), int(x2), int(y2)]


    @staticmethod
    def _fallback(w: int, h: int) -> Dict[str, Any]:
        return {
            "region_bbox": [0, 0, w, h],
            "target_rows": ["ALL"],
            "target_cols": ["ALL"],
            "confidence": 0.0,
            "routing_method": "fallback_full",
        }

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from graph_src.generation.qwen_qa import TableQAGenerator
from graph_src.structure.table_detector import TableStructureDetector, BBox


STRUCTURAL_EXTRACTION_PROMPT = """Analyze this table image and provide a structured description.

Output ONLY valid JSON in this exact format:
{
  "title": "<table title or topic if visible, else 'Unknown'>",
  "headers": ["<col1>", "<col2>", "..."],
  "row_labels": ["<row1>", "<row2>", "..."],
  "num_rows": <integer>,
  "num_cols": <integer>,
  "data_type": "<numeric|text|mixed>",
  "data_summary": "<1 sentence describing what data is in the table>"
}

IMPORTANT:
- "headers" = column header names only, left to right
- "row_labels" = first cell of each DATA row only, NOT the header row itself
- "num_rows" and "num_cols" must exactly match what you count in the image
- Do not include any text outside the JSON."""


class TableStructuralIndexBuilder:

    def __init__(
        self,
        vlm_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        detector_device: str = None,
        confidence_threshold: float = 0.5,
    ):
        detector_device = detector_device or device

        print("[StructuralIndex] Loading Table Transformer...")
        self.detector = TableStructureDetector(
            device=detector_device,
            confidence_threshold=confidence_threshold,
        )

        print(f"[StructuralIndex] Loading VLM: {vlm_model_path}")
        self.vlm = TableQAGenerator(
            model_path=vlm_model_path,
            device=device,
            load_in_4bit=load_in_4bit,
        )


    def build_structural_map(self, image: Image.Image, image_stem: str) -> Dict[str, Any]:
        w, h = image.size

        try:
            structure = self.detector.detect(image)
        except Exception as exc:
            print(f"  [WARN] Detector failed for {image_stem}: {exc}")
            structure = {"rows": [], "columns": [], "headers": []}

        try:
            raw_output = self.vlm.answer_with_patches(
                query=STRUCTURAL_EXTRACTION_PROMPT,
                patches=[image],
                max_new_tokens=512,
                temperature=0.0,
            )
            vlm_map = self._parse_vlm_output(raw_output)
        except Exception as exc:
            print(f"  [WARN] VLM extraction failed for {image_stem}: {exc}")
            vlm_map = self._default_map()

        merged = self._merge_detector_bboxes(vlm_map, structure, (w, h))
        merged["image_stem"] = image_stem
        merged["image_size"] = [w, h]
        return merged


    def _merge_detector_bboxes(self, vlm_map, structure, image_size):
        w, h = image_size
        regions = {}
        regions["full_table"] = {"bbox": [0, 0, w, h]}

        all_rows    = structure.get("rows", [])
        headers_det = structure.get("headers", [])

        header_y2 = 0
        if headers_det:
            hdr = headers_det[0]
            header_bbox = [int(hdr.x1), int(hdr.y1), int(hdr.x2), int(hdr.y2)]
            header_y2 = int(hdr.y2)
            regions["header_row"] = {"bbox": header_bbox}
        elif all_rows:
            first = all_rows[0]
            header_bbox = [int(first.x1), int(first.y1),
                           int(first.x2), int(first.y2)]
            header_y2 = int(first.y2)
            regions["header_row"] = {"bbox": header_bbox}

        data_rows = [r for r in all_rows if int(r.y1) >= header_y2 - 5]

        row_labels = vlm_map.get("row_labels", [])
        for i, row_bbox in enumerate(data_rows):
            label = row_labels[i] if i < len(row_labels) else f"row_{i}"
            regions[f"row_{i}"] = {
                "bbox": [int(row_bbox.x1), int(row_bbox.y1),
                         int(row_bbox.x2), int(row_bbox.y2)],
                "label": str(label),
            }

        det_cols = structure.get("columns", [])
        col_headers_vlm = list(vlm_map.get("headers", []))

        if abs(len(col_headers_vlm) - len(det_cols)) > 1:
            col_headers_vlm = []

        for i, col_bbox in enumerate(det_cols):
            label = col_headers_vlm[i] if i < len(col_headers_vlm) else f"col_{i}"
            regions[f"col_{i}"] = {
                "bbox": [int(col_bbox.x1), int(col_bbox.y1),
                         int(col_bbox.x2), int(col_bbox.y2)],
                "label": str(label),
            }

        vlm_map["regions"] = regions
        if vlm_map.get("num_rows", 0) == 0:
            vlm_map["num_rows"] = len(data_rows)
        if vlm_map.get("num_cols", 0) == 0:
            vlm_map["num_cols"] = len(det_cols)
        return vlm_map


    def index_directory(
        self,
        image_dir: str,
        output_dir: str = "data/indices/structural_maps",
        skip_existing: bool = True,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = sorted(
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in extensions
        )
        print(f"[StructuralIndex] Found {len(image_paths)} images in {image_dir}")

        processed, skipped, failed = 0, 0, 0
        log_path = os.path.join(output_dir, "index_log.jsonl")

        for img_path in image_paths:
            stem = img_path.stem
            out_path = os.path.join(output_dir, f"{stem}.json")

            if skip_existing and os.path.exists(out_path):
                skipped += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                smap = self.build_structural_map(image, stem)

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(smap, f, indent=2, ensure_ascii=False)

                processed += 1
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({
                        "image_stem": stem, "status": "ok",
                        "n_data_rows": len([k for k in smap.get("regions", {}) if k.startswith("row_")]),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")

                if processed % 50 == 0:
                    print(f"  progress: {processed} done, {skipped} skipped, {failed} failed")
            except Exception as exc:
                failed += 1
                print(f"  [ERROR] {stem}: {exc}")
                traceback.print_exc()
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({
                        "image_stem": stem, "status": "error",
                        "error": str(exc),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")

        print(f"[StructuralIndex] Done. processed={processed}  skipped={skipped}  failed={failed}")


    @staticmethod
    def load_map(map_path: str) -> Dict[str, Any]:
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_all_maps(maps_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load all maps keyed by image_stem. Searches recursively."""
        maps = {}
        maps_dir_p = Path(maps_dir)
        if not maps_dir_p.is_dir():
            return maps
        for p in maps_dir_p.rglob("*.json"):
            if p.suffix == ".jsonl" or p.name.endswith(".jsonl"):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                maps[p.stem] = data
            except Exception:
                continue
        return maps


    def _parse_vlm_output(self, raw: str) -> Dict[str, Any]:

        if not raw or not raw.strip():
            return self._default_map()

        text = raw.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return self._sanitize_map(obj)
        except json.JSONDecodeError:
            pass

        code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if code_block:
            try:
                obj = json.loads(code_block.group(1).strip())
                if isinstance(obj, dict):
                    return self._sanitize_map(obj)
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                obj = json.loads(brace_match.group(0))
                if isinstance(obj, dict):
                    return self._sanitize_map(obj)
            except json.JSONDecodeError:
                pass

        result = self._default_map()
        headers_m = re.search(r'"headers"\s*:\s*\[([^\]]*)\]', text)
        if headers_m:
            result["headers"] = [
                h.strip().strip('"').strip("'")
                for h in headers_m.group(1).split(",") if h.strip()
            ]
        rows_m = re.search(r'"row_labels"\s*:\s*\[([^\]]*)\]', text)
        if rows_m:
            result["row_labels"] = [
                r.strip().strip('"').strip("'")
                for r in rows_m.group(1).split(",") if r.strip()
            ]
        title_m = re.search(r'"title"\s*:\s*"([^"]*)"', text)
        if title_m:
            result["title"] = title_m.group(1)
        summary_m = re.search(r'"data_summary"\s*:\s*"([^"]*)"', text)
        if summary_m:
            result["data_summary"] = summary_m.group(1)
        return result

    @staticmethod
    def _default_map() -> Dict[str, Any]:
        return {
            "title": "Unknown", "headers": [], "row_labels": [],
            "num_rows": 0, "num_cols": 0, "data_type": "mixed",
            "data_summary": "",
        }

    @staticmethod
    def _sanitize_map(d: Dict) -> Dict[str, Any]:
        defaults = TableStructuralIndexBuilder._default_map()
        for k, v in defaults.items():
            if k not in d:
                d[k] = v
        for list_key in ("headers", "row_labels"):
            if not isinstance(d.get(list_key), list):
                d[list_key] = []
            d[list_key] = [str(x) for x in d[list_key]]
        for int_key in ("num_rows", "num_cols"):
            try:
                d[int_key] = int(d[int_key])
            except (ValueError, TypeError):
                d[int_key] = 0
        return d

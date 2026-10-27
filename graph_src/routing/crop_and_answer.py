import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_src.generation.qwen_qa import TableQAGenerator
from graph_src.routing.structural_router import StructuralRouter
from graph_src.routing.base_arithmetic_solver import ArithmeticSolver


class GraphCropAndAnswer:

    def __init__(
        self,
        vlm_generator: Optional[TableQAGenerator] = None,
        vlm_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        use_arithmetic: bool = True,
        max_images_in_context: int = 5,
        use_transcription: bool = False,
    ):
        if vlm_generator is not None:
            self.vlm = vlm_generator
        else:
            self.vlm = TableQAGenerator(
                model_path=vlm_model_path,
                device=device,
                load_in_4bit=load_in_4bit,
            )
        self.use_arithmetic = use_arithmetic
        self.max_images = max_images_in_context
        self.use_transcription = use_transcription
        self._structural_router = StructuralRouter()
        self._arith_solver = (
            ArithmeticSolver(self.vlm) if use_arithmetic else None
        )

    def answer(
        self,
        question: str,
        ranked_tables: List[Dict[str, Any]],
        maps_dir: str,
        images_dir: str,
        analysis: Dict[str, Any],
        domain: str = "",
        primary_image: Optional[Image.Image] = None,
        primary_stem: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not ranked_tables:
            return {
                "answer": "",
                "method": "no_tables",
                "tables_used": [],
                "assembly_log": [],
            }


        context = self._assemble_context(
            question, ranked_tables, maps_dir, images_dir, analysis
        )


        needs_arith = analysis.get("needs_arithmetic", False)
        needs_multi = analysis.get("needs_multi_table", False)
        all_crops = context["crops"]
        all_full = context["full_images"]


        if (
            self.use_arithmetic
            and needs_arith
            and needs_multi
            and len(all_crops) > 1
            and domain not in ("vtabfact",)
        ):
            try:
                result = self._multi_node_pot(
                    question, all_crops, all_full, context
                )
                if result is not None:
                    return {
                        "answer": result["answer"],
                        "method": "multi_node_pot",
                        "tables_used": [
                            t["image_stem"] for t in ranked_tables
                        ],
                        "assembly_log": context["assembly_log"],
                    }
            except Exception:
                pass  

        if (
            self.use_arithmetic
            and needs_arith
            and domain not in ("vtabfact",)
        ):
            try:
                primary_crop = all_crops[0] if all_crops else None
                primary_full = all_full[0] if all_full else None
                if primary_full and primary_crop:
                    bbox = context["assembly_log"][0].get(
                        "bbox", [0, 0, primary_full.width, primary_full.height]
                    ) if context["assembly_log"] else [
                        0, 0, primary_full.width, primary_full.height
                    ]
                    result = self._arith_solver.solve(
                        question=question,
                        full_image=primary_full,
                        crop_bbox=bbox,
                    )
                    if result is not None:
                        return {
                            "answer": result["answer"],
                            "method": "single_pot",
                            "tables_used": [ranked_tables[0]["image_stem"]],
                            "assembly_log": context["assembly_log"],
                        }
            except Exception:
                pass  

        prompt = self._build_graph_aware_prompt(
            question, context, analysis, domain
        )


        images = self._compose_image_list(all_crops, all_full)

        if len(images) > 1:
      
            try:
                raw = self.vlm.answer_with_patches(
                    query=prompt,
                    patches=images,
                    max_new_tokens=128,
                )
                method = "multi_patch_vlm"
            except Exception:
          
                raw = self.vlm.answer_with_patches(
                    query=prompt,
                    patches=[images[0]],
                    max_new_tokens=128,
                )
                method = "single_patch_fallback"
        elif images:
            raw = self.vlm.answer_with_patches(
                query=prompt,
                patches=[images[0]],
                max_new_tokens=128,
            )
            method = "single_patch_vlm"
        else:
            return {
                "answer": "",
                "method": "no_images",
                "tables_used": [],
                "assembly_log": context["assembly_log"],
            }

        return {
            "answer": self._normalize_answer(raw),
            "method": method,
            "tables_used": [t["image_stem"] for t in ranked_tables],
            "assembly_log": context["assembly_log"],
        }


    def answer_single(
        self,
        question: str,
        full_image: Image.Image,
        crop_bbox: List[int],
        domain: str = "",
        upscale_factor: float = 2.0,
    ) -> str:

        w, h = full_image.size
        bx1, by1, bx2, by2 = crop_bbox
        crop_area = max(1, (bx2 - bx1) * (by2 - by1))
        img_area = max(1, w * h)
        use_crop = (crop_area / img_area) < 0.60


        if self.use_arithmetic and domain not in ("vtabfact",):
            try:
                arith = self._arith_solver.solve(
                    question=question,
                    full_image=full_image,
                    crop_bbox=crop_bbox,
                    upscale_factor=upscale_factor,
                )
                if arith is not None:
                    return arith["answer"]
            except Exception:
                pass

        prompt = self._build_single_prompt(question, domain)

        if use_crop:
            cropped = self._crop_and_upscale(
                full_image, crop_bbox, upscale_factor
            )
            try:
                raw = self.vlm.answer_with_context(
                    query=prompt,
                    full_image=full_image,
                    patches=[cropped],
                    max_new_tokens=128,
                )
            except Exception:
                raw = self.vlm.answer_with_patches(
                    query=prompt,
                    patches=[full_image],
                    max_new_tokens=128,
                )
        else:
            raw = self.vlm.answer_with_patches(
                query=prompt,
                patches=[full_image],
                max_new_tokens=128,
            )

        return self._normalize_answer(raw)


    def _assemble_context(
        self,
        question: str,
        ranked_tables: List[Dict[str, Any]],
        maps_dir: str,
        images_dir: str,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:

        crops = []
        full_images = []
        context_parts = []
        assembly_log = []

        for i, table_info in enumerate(
            ranked_tables[:self.max_images]
        ):
            image_stem = table_info["image_stem"]

            image_path = self._find_image(images_dir, image_stem)
            if not image_path:
                assembly_log.append({
                    "table": image_stem,
                    "status": "image_not_found",
                })
                continue

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                assembly_log.append({
                    "table": image_stem,
                    "status": f"image_error: {e}",
                })
                continue

            map_path = Path(maps_dir) / f"{image_stem}.json"
            if not map_path.exists():
 
                for subdir in Path(maps_dir).iterdir():
                    if subdir.is_dir():
                        candidate = subdir / f"{image_stem}.json"
                        if candidate.exists():
                            map_path = candidate
                            break
            if map_path.exists():
                try:
                    with open(map_path, "r", encoding="utf-8") as f:
                        struct_map = json.load(f)
                except (json.JSONDecodeError, OSError):
                    struct_map = None
            else:
                struct_map = None

            if struct_map:
   
                try:
                    routing = self._structural_router.route(
                        question, struct_map, img
                    )
                except Exception:
                    routing = StructuralRouter._fallback(img.width, img.height)

                bbox = routing.get(
                    "region_bbox", [0, 0, img.width, img.height]
                )

  
                crop = self._crop_and_upscale(img, bbox, upscale_factor=2.0)
                crops.append(crop)

    
                if i == 0:
                    full_images.append(img)

 
                title = struct_map.get("title", "Unknown")
                headers = ", ".join(struct_map.get("headers", [])[:6])
                context_parts.append(
                    f"Table {i + 1}: '{title}' | Columns: {headers}"
                )

                assembly_log.append({
                    "table": image_stem,
                    "relevance": table_info.get("score", 0),
                    "reason": table_info.get("reason", ""),
                    "bbox": bbox,
                    "routing_method": routing.get("routing_method", ""),
                    "status": "ok",
                })
            else:
  
                if i == 0:
                    full_images.append(img)
                    crops.append(img)
                context_parts.append(
                    f"Table {i + 1}: '{image_stem}' (no structure map)"
                )
                assembly_log.append({
                    "table": image_stem,
                    "status": "no_structural_map",
                })

        return {
            "crops": crops,
            "full_images": full_images,
            "context_description": "\n".join(context_parts),
            "assembly_log": assembly_log,
        }


    @staticmethod
    def _build_graph_aware_prompt(
        question: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any],
        domain: str = "",
    ) -> str:

        needs_multi = analysis.get("needs_multi_table", False)
        desc = context.get("context_description", "")
        table_lines = [l for l in desc.strip().split("\n") if l.strip()]

        if needs_multi and len(table_lines) > 1:

            comparison = analysis.get("comparison_type", "none")
            if comparison == "temporal":
                years = analysis.get("years_mentioned", [])
                steps = []
                for idx, year in enumerate(years):
                    steps.append(
                        f"Step {idx + 1}: Find the relevant value "
                        f"in the table for {year}"
                    )
                steps.append(
                    f"Step {len(years) + 1}: Compute the final answer"
                )
                step_text = "\n".join(steps)

                prompt = (
                    f"You are given {len(table_lines)} related tables:\n"
                    f"{desc}\n\n"
                    f"Using information from BOTH tables, answer step by step:\n"
                    f"{step_text}\n\n"
                    f"Question: {question}\n"
                    f"Final Answer:"
                )
            else:
                prompt = (
                    f"You are given {len(table_lines)} related tables:\n"
                    f"{desc}\n\n"
                    f"Using information from the tables above, "
                    f"answer the following question.\n"
                    f"Provide the shortest possible exact answer.\n\n"
                    f"Question: {question}\n"
                    f"Answer:"
                )
        else:

            prompt = GraphCropAndAnswer._build_single_prompt(
                question, domain
            )

            if table_lines:
                prompt = f"Context: {table_lines[0]}\n\n{prompt}"

        return prompt

    @staticmethod
    def _build_single_prompt(question: str, domain: str = "") -> str:

        if domain == "vtabfact":
            return (
                "Is the following statement True or False based on the table? "
                "Answer with True or False only.\n"
                f"Statement: {question}"
            )
        if domain == "fintabnetqa":
            return (
                "Answer the following question about this financial table. "
                "Give only the final value, written exactly as it appears in the table. "
                "Keep the currency symbol and percent sign. If the table or its header "
                "states the amounts are in thousands, millions or billions, append that "
                "scale word to the value (e.g. \"$894.4 million\"). "
                "Write negative numbers in parentheses, e.g. \"$(16.7) million\".\n"
                f"Question: {question}"
            )
        return question


    def _multi_node_pot(
        self,
        question: str,
        crops: List[Image.Image],
        full_images: List[Image.Image],
        context: Dict[str, Any],
    ) -> Optional[dict]:

        if not self._arith_solver:
            return None

        desc = context.get("context_description", "")

        multi_pot_prompt = (
            f"You are given multiple table images.\n{desc}\n\n"
            "Look at ALL the table images carefully. "
            "Write a short Python program that:\n"
            "1. Reads the relevant numeric values from each table.\n"
            "2. Computes the answer using values from MULTIPLE tables.\n"
            "3. Ends with a single print() statement.\n\n"
            "Rules:\n"
            "- No imports allowed.\n"
            "- Variable names: val_table1, val_table2, result, etc.\n"
            "- Strip commas, $, % from numbers.\n"
            "- If no arithmetic needed, print(\"NA\").\n\n"
            f"Question: {question}\n\n"
            "Python code:"
        )


        images = crops[:3]

        try:
            code_raw = self.vlm.answer_with_patches(
                query=multi_pot_prompt,
                patches=images,
                max_new_tokens=256,
            )
        except Exception:
            return None


        from graph_src.routing.base_arithmetic_solver import (
            _safe_exec,
            _pretty_number,
            _strip_preamble,
        )

        numeric_result = _safe_exec(code_raw)
        if numeric_result is None:
            return None


        format_prompt = (
            f"The computed numeric answer to the question is: "
            f"{_pretty_number(numeric_result)}\n\n"
            f"Question: {question}\n\n"
            "Write the final answer exactly as it would appear — "
            "include units, currency symbols ($, %), and formatting. "
            "Give only the answer, no explanation."
        )
        try:
            formatted = self.vlm.answer_with_patches(
                query=format_prompt,
                patches=[images[0]],
                max_new_tokens=64,
            )
        except Exception:
            formatted = str(_pretty_number(numeric_result))

        formatted = _strip_preamble(formatted)

        return {
            "answer": formatted,
            "computed": numeric_result,
            "code": code_raw,
        }


    def _compose_image_list(
        self,
        crops: List[Image.Image],
        full_images: List[Image.Image],
    ) -> List[Image.Image]:

        images = []


        if crops:
            images.append(crops[0])


        if full_images:
            images.append(full_images[0])


        for crop in crops[1:]:
            if len(images) >= self.max_images:
                break
            images.append(crop)

        return images

    @staticmethod
    def _crop_and_upscale(
        image: Image.Image,
        bbox: List[int],
        upscale_factor: float,
        padding: int = 20,
    ) -> Image.Image:

        w, h = image.size
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        cropped = image.crop((x1, y1, x2, y2))
        if upscale_factor > 1.0:
            new_w = int(cropped.width * upscale_factor)
            new_h = int(cropped.height * upscale_factor)
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        return cropped

    @staticmethod
    def _find_image(images_dir: str, image_stem: str) -> Optional[str]:

        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:

            path = Path(images_dir) / f"{image_stem}{ext}"
            if path.exists():
                return str(path)

            for subdir in Path(images_dir).iterdir():
                if subdir.is_dir():
                    path = subdir / f"{image_stem}{ext}"
                    if path.exists():
                        return str(path)
        return None

    @staticmethod
    def _normalize_answer(answer: str) -> str:

        if not answer:
            return ""
        answer = re.sub(
            r"^(the answer is|answer:|final answer:|based on the table,?|"
            r"according to the table,?|from the table,?|"
            r"looking at the table,?|the table shows,?)\s*",
            "", answer.strip(), flags=re.IGNORECASE,
        )
        answer = answer.strip().strip(".,;:")
        answer = re.sub(r"\s+", " ", answer)


        _trunc = re.search(
            r'^(.{5,80}?)(?:\.\s+[A-Z]|\.\s*$|,\s+(?:which|where|as|since|because|however))',
            answer
        )
        if _trunc:
            answer = _trunc.group(1).strip().rstrip(',').rstrip('.')


        answer = re.sub(r'\$\s+', '$', answer)


        answer = re.sub(r'(\d)\u2013(\d)', r'\1-\2', answer)
        answer = re.sub(r'(\d)\u2014(\d)', r'\1-\2', answer)


        if '\n' in answer and not re.search(r'[.?!]', answer):
            parts = [p.strip() for p in answer.split('\n') if p.strip()]
            if len(parts) > 1:
                answer = ', '.join(parts)

        return answer.strip()

import re
import math
from typing import Optional

from PIL import Image

from graph_src.generation.qwen_qa import TableQAGenerator


POT_PROMPT = """Look at the table image carefully. Write a short Python program that:
1. Reads the relevant numeric values directly from the table as variables.
2. Computes the answer to the question using those variables.
3. Ends with a single print() statement that outputs ONLY the numeric result.

Rules:
- No imports allowed.
- Variable names must be simple (val_a, val_b, result, etc.).
- If a value has commas (1,234) write it as 1234. If it has $ or % strip them.
- The final print() must output a single number, nothing else.
- If the question cannot be answered with arithmetic, print("NA").

Question: {question}

Python code:"""


FORMAT_PROMPT = """The computed numeric answer to the question is: {result}

Question: {question}

Now write the final answer exactly as it would appear in the table --
include the correct units, currency symbols ($, %), formatting, and
any context needed. Give only the answer, no explanation."""


def _safe_exec(code: str) -> Optional[float]:

    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)
    code = code.strip()

    if not code:
        return None

    if re.search(r"\bimport\b", code):
        return None

    captured = []

    def _capture_print(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    safe_globals = {
        "__builtins__": {
            "print": _capture_print,
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "range": range,
            "int": int, "float": float, "str": str,
            "True": True, "False": False, "None": None,
        }
    }
    safe_locals = {}

    try:
        exec(compile(code, "<pot>", "exec"), safe_globals, safe_locals)
    except Exception:
        return None

    if not captured:
        return None

    output = captured[-1].strip()

    if output.upper() == "NA":
        return None

    clean = re.sub(r"[,$%]", "", output).strip()
    clean = re.sub(r"\s*(thousand|million|billion)s?\b", "", clean,
                   flags=re.IGNORECASE).strip()
    try:
        return float(clean)
    except ValueError:
        return None


class ArithmeticSolver:

    def __init__(self, vlm: TableQAGenerator):
        self.vlm = vlm

    _ARITH_TRIGGERS = [
        r"what is the (total|sum|difference|average|mean|ratio|percent.?change|% change|net|combined|aggregate)",
        r"how much (more|less|greater|smaller|higher|lower)",
        r"by how (much|many)",
        r"what (was|were|is) the (total|sum|difference|change|increase|decrease|growth|decline|net)",
        r"calculate",
        r"what percent(age)? (of|is|are|was|were).*(total|all|combined|overall)",
    ]

    _ARITH_BLOCKERS = [
        r"^\s*which",
        r"^\s*who",
        r"^\s*what (type|kind|form|category|group|class|name|label|term|word|drug|gene|protein|country|city|region|disease|method|technique|study|author|compound|element|enzyme|treatment)",
        r"in which",
        r"^\s*when",
        r"^\s*where",
        r"what does .* stand for",
        r"what is the name",
        r"what is the (most|least) common",
        r"what is the (highest|lowest|largest|smallest|best|worst) .*(type|name|drug|gene|study|group|category|method|country|city|author|compound)",
    ]

    @classmethod
    def _needs_arithmetic(cls, question: str) -> bool:
        q = question.lower().strip()
        for pat in cls._ARITH_BLOCKERS:
            if re.search(pat, q):
                return False
        for pat in cls._ARITH_TRIGGERS:
            if re.search(pat, q):
                return True
        return False

    def solve(
        self,
        question: str,
        full_image: Image.Image,
        crop_bbox: list,
        upscale_factor: float = 2.0,
        padding: int = 20,
    ) -> Optional[dict]:

        if not self._needs_arithmetic(question):
            return None

        cropped = self._crop_and_upscale(
            full_image, crop_bbox, upscale_factor, padding
        )

        pot_prompt = POT_PROMPT.format(question=question)
        try:
            code_raw = self.vlm.answer_with_context(
                query=pot_prompt,
                full_image=full_image,
                patches=[cropped],
                max_new_tokens=256,
            )
        except Exception:
            try:
                code_raw = self.vlm.answer_with_patches(
                    query=pot_prompt,
                    patches=[cropped],
                    max_new_tokens=256,
                )
            except Exception:
                return None

        numeric_result = _safe_exec(code_raw)
        if numeric_result is None:
            return None

        format_prompt = FORMAT_PROMPT.format(
            result=_pretty_number(numeric_result),
            question=question,
        )
        try:
            formatted = self.vlm.answer_with_patches(
                query=format_prompt,
                patches=[full_image],
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

    @staticmethod
    def _crop_and_upscale(
        image: Image.Image,
        bbox: list,
        upscale_factor: float,
        padding: int,
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


def _pretty_number(val: float) -> str:
    if val == int(val) and abs(val) < 1e12:
        return str(int(val))
    return "{:.4g}".format(val)


def _strip_preamble(text: str) -> str:
    text = re.sub(
        r"^(the answer is|answer:|final answer:|based on the table,?|"
        r"according to the table,?|the computed answer is[^.]*\.\s*)",
        "", text.strip(), flags=re.IGNORECASE,
    )
    return text.strip().strip(".,;:")

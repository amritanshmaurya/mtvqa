import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class QuestionType(str, Enum):
    SINGLE_TABLE = "single_table"
    MULTI_TABLE = "multi_table"
    TEMPORAL = "temporal"
    ARITHMETIC = "arithmetic"
    ENTITY_LOOKUP = "entity_lookup"



_MULTI_TABLE_PATTERNS = [
    r"\bcompared?\s+to\b",
    r"\bchange\s+from\b",
    r"\bdifference\s+between\b",
    r"\byear[- ]over[- ]year\b",
    r"\bacross\s+(all\s+)?tables?\b",
    r"\bcombined\b",
    r"\btotal\s+across\b",
    r"\bjoin(ed|ing)?\b",
    r"\brelative\s+to\b",
    r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
    r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
    r"\b(more|less|greater|higher|lower)\s+than\b.*\b(in|during|for)\b",
]


_TEMPORAL_PATTERNS = [
    r"\b%\s*change\b",
    r"\bpercent(age)?\s*(change|increase|decrease|growth|decline)\b",
    r"\bgrowth\s+(rate|from)\b",
    r"\bincrease(d)?\s+from\b",
    r"\bdecrease(d)?\s+from\b",
    r"\b(rose|fell|dropped|climbed|declined)\s+(from|by)\b",
    r"\bfrom\s+(fiscal\s+)?(year\s+)?\d{4}\b",
    r"\b\d{4}\s+to\s+\d{4}\b",
    r"\b\d{4}\s+(and|vs\.?|versus)\s+\d{4}\b",
]


_ARITH_TRIGGERS = [
    r"\bwhat\s+is\s+the\s+(total|sum|difference|average|mean|ratio)\b",
    r"\bpercent(age)?[\s]*(change|of)\b",
    r"\b%\s*change\b",
    r"\bhow\s+much\s+(more|less|greater|smaller|higher|lower)\b",
    r"\bby\s+how\s+(much|many)\b",
    r"\bwhat\s+(was|were|is)\s+the\s+(total|sum|difference|change|increase|decrease|growth|decline|net)\b",
    r"\bcalculate\b",
    r"\bwhat\s+percent(age)?\s+(of|is|are|was|were)\b",
    r"\bwhat\s+proportion\b",
    r"\bwhat\s+fraction\b",
    r"\bnet\s+(change|income|revenue|profit|loss)\b",
    r"\b(combined|aggregate|cumulative)\s+(total|sum|value)\b",
]


_ARITH_BLOCKERS = [
    r"^\s*which\b",
    r"^\s*who\b",
    r"^\s*what\s+(type|kind|form|category|group|class|name|label|term|word|drug|gene|protein|country|city|region|disease|method|technique|study|author|compound|element|enzyme|treatment)\b",
    r"\bin\s+which\b",
    r"^\s*when\b",
    r"^\s*where\b",
    r"\bwhat\s+does\s+.*\s+stand\s+for\b",
    r"\bwhat\s+is\s+the\s+name\b",
    r"\bwhat\s+is\s+the\s+(most|least)\s+common\b",
    r"\btrue\s+or\s+false\b",
    r"\bis\s+(it|this|the)\s+(true|false|correct)\b",
]


_YEAR_PATTERN = re.compile(
    r"\b(20\d{2}|19\d{2}|Q[1-4]|FY\d{2,4}|H[12])\b", re.IGNORECASE
)


_METRIC_PATTERNS = [
    r"\b(net\s+revenue|gross\s+profit|operating\s+income|net\s+income|"
    r"total\s+assets|total\s+liabilities|total\s+equity|"
    r"operating\s+expenses?|capital\s+expenditures?|"
    r"free\s+cash\s+flow|earnings\s+per\s+share|"
    r"cost\s+of\s+(goods\s+sold|revenue|sales)|"
    r"research\s+and\s+development|"
    r"selling.*general.*administrative|"
    r"interest\s+expense|income\s+tax|"
    r"gross\s+margin|operating\s+margin|net\s+margin|"
    r"return\s+on\s+(assets|equity|investment)|"
    r"debt\s+to\s+equity|current\s+ratio)\b",
    # Single-word metrics
    r"\b(revenue|income|profit|loss|cost|expense|margin|"
    r"assets|liabilities|equity|cash|debt|sales|ebitda|"
    r"earnings|dividend|tax|depreciation|amortization|"
    r"interest|capital|investment|return|growth|"
    r"yield|efficiency|concentration|activity|output)\b",
]


_FULL_TABLE_PATTERNS = [
    r"\bhow\s+many\b", r"\btotal\b", r"\bsum\b", r"\baverage\b",
    r"\bmean\b", r"\bcount\b", r"\ball\s+rows\b", r"\beach\s+row\b",
    r"\bevery\s+row\b", r"\blist\s+all\b", r"\benumerate\b",
    r"\brank\b", r"\boverall\b", r"\bacross\s+all\b",
]
_SUPERLATIVE_PATTERNS = [
    r"\bhighest\b", r"\blowest\b", r"\blargest\b", r"\bsmallest\b",
    r"\bmaximum\b", r"\bminimum\b", r"\bmost\b", r"\bfewest\b",
    r"\bbest\b", r"\bworst\b", r"\bgreatest\b",
]


class GraphRouter:

    def classify(self, question: str) -> Dict[str, Any]:

        q = question.strip()
        q_lower = q.lower()

        years = self._extract_years(q)
        metrics = self._extract_metrics(q)
        entities = self._extract_entities(q, years, metrics)

        needs_arith = self._needs_arithmetic(q_lower)
        needs_multi = self._needs_multi_table(q_lower, years, metrics)
        is_temporal = self._is_temporal(q_lower, years)
        needs_full = self._needs_full_table(q_lower)


        if is_temporal and len(years) >= 2:
            qtype = QuestionType.TEMPORAL
            comparison_type = "temporal"
        elif needs_multi:
            qtype = QuestionType.MULTI_TABLE
            comparison_type = "cross_table"
        elif needs_arith:
            qtype = QuestionType.ARITHMETIC
            comparison_type = "none"
        else:
            qtype = QuestionType.SINGLE_TABLE
            comparison_type = "none"

        return {
            "question_type": qtype,
            "needs_multi_table": needs_multi or is_temporal,
            "needs_arithmetic": needs_arith,
            "needs_full_table": needs_full,
            "years_mentioned": years,
            "metrics_mentioned": metrics,
            "entities": entities,
            "comparison_type": comparison_type,
        }


    @staticmethod
    def _extract_years(question: str) -> List[str]:

        return _YEAR_PATTERN.findall(question)

    @staticmethod
    def _extract_metrics(question: str) -> List[str]:

        metrics = []
        q_lower = question.lower()
        for pattern in _METRIC_PATTERNS:
            matches = re.findall(pattern, q_lower)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0] 
                m = m.strip()
                if m and m not in metrics:
                    metrics.append(m)
        return metrics

    @staticmethod
    def _extract_entities(
        question: str,
        years: List[str],
        metrics: List[str],
    ) -> Set[str]:

        entities: Set[str] = set()

        for y in years:
            entities.add(y)
        for m in metrics:
            entities.add(m.lower())


        stop_words = {
            "the", "a", "an", "of", "in", "for", "and", "or", "to", "by",
            "at", "as", "is", "are", "was", "were", "be", "been", "being",
            "what", "which", "how", "much", "many", "from", "between",
            "this", "that", "these", "those", "it", "its", "not", "no",
            "with", "than", "but", "about", "into", "did", "do", "does",
            "will", "would", "could", "should", "can", "may", "might",
        }
        for token in question.lower().split():
            clean = re.sub(r"[^\w]", "", token)
            if len(clean) > 2 and clean not in stop_words:
                entities.add(clean)

        return entities


    @staticmethod
    def _needs_multi_table(
        q_lower: str,
        years: List[str],
        metrics: List[str],
    ) -> bool:


        for pattern in _MULTI_TABLE_PATTERNS:
            if re.search(pattern, q_lower):
                return True


        if len(years) >= 2:
            comp_keywords = [
                "change", "difference", "compared", "growth",
                "increase", "decrease", "from", "between",
            ]
            if any(kw in q_lower for kw in comp_keywords):
                return True

        return False

    @staticmethod
    def _is_temporal(q_lower: str, years: List[str]) -> bool:

        if len(years) < 2:
            return False
        for pattern in _TEMPORAL_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        return False

    @staticmethod
    def _needs_arithmetic(q_lower: str) -> bool:

        for pattern in _ARITH_BLOCKERS:
            if re.search(pattern, q_lower):
                return False

        for pattern in _ARITH_TRIGGERS:
            if re.search(pattern, q_lower):
                return True
        return False

    @staticmethod
    def _needs_full_table(q_lower: str) -> bool:
        return any(
            re.search(p, q_lower)
            for p in _FULL_TABLE_PATTERNS + _SUPERLATIVE_PATTERNS
        )

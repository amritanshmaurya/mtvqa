import re
import math
from typing import Dict, List, Optional, Set, Tuple

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "by",
    "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "shall",
    "would", "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "it", "its", "not", "no", "nor",
    "with", "from", "than", "but", "about", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "under", "over", "out", "up", "down", "on", "off", "then",
    "once", "here", "there", "when", "where", "why", "how",
    "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "so", "very",
    "table", "figure", "note", "notes", "source", "total",
})

_YEAR_PATTERN = re.compile(
    r"\b(20\d{2}|19\d{2}|Q[1-4]|FY\d{2,4}|H[12]|1H|2H)\b", re.IGNORECASE
)
_CURRENCY_CHARS = "$\u20ac\u00a3"  # $ € £
_NUMBER_PATTERN = re.compile(
    r"[" + re.escape(_CURRENCY_CHARS) + r"]?\s*-?\d[\d,]*\.?\d*\s*[%]?"
)


_METRIC_KEYWORDS = frozenset({
    "revenue", "income", "profit", "loss", "cost", "expense", "margin",
    "assets", "liabilities", "equity", "cash", "debt", "sales", "ebitda",
    "earnings", "dividend", "tax", "depreciation", "amortization",
    "interest", "capital", "investment", "return", "growth", "rate",
    "ratio", "percentage", "share", "stock", "price", "value",
    "volume", "quantity", "amount", "balance", "flow", "net", "gross",
    "operating", "total", "average", "median", "standard", "deviation",
    "concentration", "activity", "yield", "efficiency", "output",
})


def extract_entities(structural_map: dict) -> Set[str]:

    entities: Set[str] = set()


    title = structural_map.get("title", "")
    entities.update(_YEAR_PATTERN.findall(title))
    entities.update(_normalize_terms(title.split()))


    for h in structural_map.get("headers", []):
        entities.update(_YEAR_PATTERN.findall(h))
        entities.update(_normalize_terms(h.split()))


    for r in structural_map.get("row_labels", []):
        entities.update(_YEAR_PATTERN.findall(r))
        entities.update(_normalize_terms(r.split()))


    summary = structural_map.get("data_summary", "")
    entities.update(_normalize_terms(summary.split()))


    entities = {
        e for e in entities
        if len(e) > 1 and e.lower() not in _STOP_WORDS
    }
    return entities


def extract_numeric_fingerprint(structural_map: dict) -> Dict[str, float]:

    text_blob = " ".join([
        structural_map.get("title", ""),
        structural_map.get("data_summary", ""),
        " ".join(structural_map.get("headers", [])),
        " ".join(structural_map.get("row_labels", [])),
    ])

    has_currency = bool(re.search(r"[" + re.escape(_CURRENCY_CHARS) + r"]", text_blob))
    has_percentage = "%" in text_blob or "percent" in text_blob.lower()


    numbers = _NUMBER_PATTERN.findall(text_blob)
    max_magnitude = 0
    for n in numbers:
        clean = re.sub(r"[,$%" + re.escape(_CURRENCY_CHARS) + r"\s]", "", n)
        try:
            val = abs(float(clean))
            if val > 0:
                max_magnitude = max(max_magnitude, int(math.log10(val)) + 1)
        except (ValueError, OverflowError):
            pass


    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text_blob)]
    year_range = (min(years), max(years)) if years else None

    return {
        "has_currency": has_currency,
        "has_percentage": has_percentage,
        "magnitude_order": max_magnitude,
        "year_range": year_range,
    }



def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> Tuple[float, List[str]]:

    if not set_a or not set_b:
        return 0.0, []
    shared = set_a & set_b
    union = set_a | set_b
    score = len(shared) / len(union) if union else 0.0
    return score, sorted(shared)


def numeric_correlation(fp_a: dict, fp_b: dict) -> float:

    score = 0.0
    factors = 0


    if fp_a.get("has_currency") == fp_b.get("has_currency"):
        score += 0.2
    factors += 0.2


    if fp_a.get("has_percentage") == fp_b.get("has_percentage"):
        score += 0.15
    factors += 0.15


    mag_a = fp_a.get("magnitude_order", 0)
    mag_b = fp_b.get("magnitude_order", 0)
    if mag_a > 0 and mag_b > 0 and abs(mag_a - mag_b) <= 1:
        score += 0.3
    elif mag_a > 0 and mag_b > 0:
        score += max(0, 0.3 - 0.1 * abs(mag_a - mag_b))
    factors += 0.3


    yr_a = fp_a.get("year_range")
    yr_b = fp_b.get("year_range")
    if yr_a and yr_b:
        overlap = max(0, min(yr_a[1], yr_b[1]) - max(yr_a[0], yr_b[0]) + 1)
        span = max(1, max(yr_a[1], yr_b[1]) - min(yr_a[0], yr_b[0]) + 1)
        score += 0.35 * (overlap / span)
    factors += 0.35

    return score / factors if factors > 0 else 0.0



class EntityLinker:

    def __init__(
        self,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        w_jaccard: float = 0.5,
        w_numeric: float = 0.3,
        w_embedding: float = 0.2,
    ):
        self.use_embeddings = use_embeddings
        self.w_jaccard = w_jaccard
        self.w_numeric = w_numeric
        self.w_embedding = w_embedding


        if not use_embeddings:
            total = w_jaccard + w_numeric
            self.w_jaccard = w_jaccard / total if total > 0 else 0.5
            self.w_numeric = w_numeric / total if total > 0 else 0.5
            self.w_embedding = 0.0

        self._embedder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(embedding_model)
                print(f"[EntityLinker] Loaded embedding model: {embedding_model}")
            except ImportError:
                print("[EntityLinker] sentence-transformers not installed, "
                      "falling back to string-only linking.")
                self.use_embeddings = False
                total = w_jaccard + w_numeric
                self.w_jaccard = w_jaccard / total if total > 0 else 0.5
                self.w_numeric = w_numeric / total if total > 0 else 0.5
                self.w_embedding = 0.0

    def link(
        self,
        entities_a: Set[str],
        entities_b: Set[str],
        fingerprint_a: dict,
        fingerprint_b: dict,
        text_a: str = "",
        text_b: str = "",
    ) -> Tuple[float, List[str], str]:

        j_score, shared = jaccard_similarity(entities_a, entities_b)


        n_score = numeric_correlation(fingerprint_a, fingerprint_b)

 
        e_score = 0.0
        if self.use_embeddings and self._embedder and text_a and text_b:
            e_score = self._embedding_similarity(text_a, text_b)


        combined = (
            self.w_jaccard * j_score
            + self.w_numeric * n_score
            + self.w_embedding * e_score
        )


        scores = {
            "entity_overlap": j_score,
            "numeric_corr": n_score,
            "embedding_sim": e_score,
        }
        dominant = max(scores, key=scores.get)

        return combined, shared, dominant

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:

        if not self._embedder:
            return 0.0
        try:
            embeddings = self._embedder.encode(
                [text_a, text_b], normalize_embeddings=True
            )
            sim = float(embeddings[0] @ embeddings[1])
            return max(0.0, sim)
        except Exception:
            return 0.0

    def embed_text(self, text: str):

        if not self._embedder:
            return None
        try:
            return self._embedder.encode(text, normalize_embeddings=True)
        except Exception:
            return None



def _normalize_terms(terms: List[str]) -> List[str]:

    normalized = []
    for t in terms:
        t = re.sub(r"[^\w\s]", "", t.lower().strip())
        if len(t) > 2 and t not in _STOP_WORDS:
            normalized.append(t)
    return normalized

import re
import unicodedata
from typing import Tuple


def normalize_for_match(text: str) -> str:

    if not text:
        return ""


    s = unicodedata.normalize("NFKC", str(text))

    s = s.lower()

    s = s.strip().strip("\"'`")

    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\b(a|an|the)\b", "", s)

    s = re.sub(r"\b(approximately|about|around|roughly|nearly|almost)\b", "", s)

    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\$\s+", "$", s)
    s = re.sub(r"€\s+", "€", s)
    s = re.sub(r"£\s+", "£", s)

    s = re.sub(r"(\d)\s+%", r"\1%", s)

    s = re.sub(r"\s*±\s*", "±", s)

    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)  

    s = re.sub(r"\s*[\-\u2013\u2014]\s*", "-", s)

    s = s.rstrip(".,;:")

    return s.strip()


def normalized_exact_match(prediction: str, reference: str) -> bool:

    return normalize_for_match(prediction) == normalize_for_match(reference)


def normalized_substring_match(prediction: str, reference: str) -> bool:

    norm_pred = normalize_for_match(prediction)
    norm_ref = normalize_for_match(reference)
    if not norm_ref:
        return True  
    return norm_ref in norm_pred


def compute_normalized_em(
    predictions: list, references: list
) -> float:

    if not predictions:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references)
        if normalized_exact_match(p, r)
    )
    return matches / len(predictions)


def compute_normalized_substring_acc(
    predictions: list, references: list
) -> float:

    if not predictions:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references)
        if normalized_substring_match(p, r)
    )
    return matches / len(predictions)

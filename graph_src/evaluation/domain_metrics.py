import re
import unicodedata
from math import isnan, isinf
from typing import Dict, List, Tuple


def _wtq_normalize(x: str) -> str:

    if not isinstance(x, str):
        x = str(x)
    x = ''.join(
        c for c in unicodedata.normalize('NFKD', x)
        if unicodedata.category(c) != 'Mn'
    )
    x = re.sub(r"[''´`]", "'", x)
    x = re.sub(r'["\u201c\u201d]', '"', x)
    x = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', x)
    while True:
        old_x = x
        x = re.sub(r'((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$', '', x.strip())
        x = re.sub(r'(?<!^)( \([^)]*\))*$', '', x.strip())
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    if x and x[-1] == '.':
        x = x[:-1]
    x = re.sub(r'\s+', ' ', x).lower().strip()
    return x


def _parse_number(text: str):

    try:
        return int(text)
    except (ValueError, TypeError):
        try:
            amount = float(text)
            if isnan(amount) or isinf(amount):
                return None
            return amount
        except (ValueError, TypeError):
            return None


def _parse_date(text: str):

    try:
        parts = text.lower().strip().split('-')
        if len(parts) != 3:
            return None
        year = -1 if parts[0] in ('xx', 'xxxx') else int(parts[0])
        month = -1 if parts[1] == 'xx' else int(parts[1])
        day = -1 if parts[2] == 'xx' else int(parts[2])
        if year == month == day == -1:
            return None
        if month != -1 and not (1 <= month <= 12):
            return None
        if day != -1 and not (1 <= day <= 31):
            return None
        return (year, month, day)
    except (ValueError, TypeError):
        return None


def _to_value(text: str) -> Tuple[str, object]:

    text = str(text).strip()

    amount = _parse_number(text)
    if amount is not None:
        return ('number', amount, _wtq_normalize(text))

    ymd = _parse_date(text)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return ('number', ymd[0], _wtq_normalize(text))
        return ('date', ymd, _wtq_normalize(text))

    return ('string', text, _wtq_normalize(text))


def _values_match(target_val, pred_val) -> bool:

    t_type, t_val, t_norm = target_val
    p_type, p_val, p_norm = pred_val

    if t_norm == p_norm:
        return True

    if t_type == 'number' and p_type == 'number':
        return abs(t_val - p_val) < 1e-6

    if t_type == 'date' and p_type == 'date':
        return t_val == p_val

    return False


def _tsv_unescape(text: str) -> str:

    return text.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def wtq_denotation_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:

    n_correct = 0
    n_total = len(predictions)

    for pred_str, ref_str in zip(predictions, references):
        try:
            ref_unescaped = _tsv_unescape(str(ref_str))
            ref_items = _split_answer(ref_unescaped)
            pred_items = _split_answer(str(pred_str))

            target_values = [_to_value(x) for x in ref_items]
            predicted_values = [_to_value(x) for x in pred_items]

            if _check_denotation(target_values, predicted_values):
                n_correct += 1
        except Exception:
            pass

    acc = n_correct / max(1, n_total)
    return {
        "primary_accuracy": acc,
        "denotation_accuracy": acc,
        "n_correct": n_correct,
        "n_total": n_total,
    }


def _split_answer(text: str) -> List[str]:

    text = text.strip()
    if not text:
        return [""]

    if '|' in text:
        items = [x.strip() for x in text.split('|') if x.strip()]
        if items:
            return items

    if ', ' in text and not re.match(r'^[\d,]+\.?\d*$', text):
        items = [x.strip() for x in text.split(', ') if x.strip()]
        if len(items) > 1:
            return items

    return [text]


def _check_denotation(target_values, predicted_values) -> bool:

    if len(target_values) != len(predicted_values):
        return False
    for target in target_values:
        if not any(_values_match(target, pred) for pred in predicted_values):
            return False
    return True



def _normalize_tf(answer: str) -> str:

    a = str(answer).strip().lower()
    if a in ('true', '1', 'yes', 'correct', 'supported', 'entailed'):
        return '1'
    if a in ('false', '0', 'no', 'incorrect', 'refuted', 'not supported',
             'not entailed', 'wrong'):
        return '0'
    if a.startswith('true') or a.startswith('yes'):
        return '1'
    if a.startswith('false') or a.startswith('no'):
        return '0'
    if 'true' in a and 'false' not in a:
        return '1'
    if 'false' in a and 'true' not in a:
        return '0'
    return a


def vtabfact_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:

    n_correct = 0
    n_total = len(predictions)
    n_true_correct = 0
    n_true_total = 0
    n_false_correct = 0
    n_false_total = 0

    for pred_str, ref_str in zip(predictions, references):
        pred_norm = _normalize_tf(pred_str)
        ref_norm = _normalize_tf(ref_str)

        match = (pred_norm == ref_norm)
        if match:
            n_correct += 1

        if ref_norm == '1':
            n_true_total += 1
            if match:
                n_true_correct += 1
        elif ref_norm == '0':
            n_false_total += 1
            if match:
                n_false_correct += 1

    acc = n_correct / max(1, n_total)
    return {
        "primary_accuracy": acc,
        "classification_accuracy": acc,
        "true_accuracy": n_true_correct / max(1, n_true_total),
        "false_accuracy": n_false_correct / max(1, n_false_total),
        "n_correct": n_correct,
        "n_total": n_total,
    }



_FINTAB_UNIT_WORDS = [
    'dollar', 'gallon', 'square foot', 'square feet',
    'share', 'mmbtu', 'mbtu', 'mbpd', 'mbbls', 'mboe',
    'unit', 'gwh', 'mwh', 'kwh', 'year', 'mmcf', 'mcf', 'mile',
]

_FINTAB_UNIT_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in _FINTAB_UNIT_WORDS) + r')s?\b',
    re.IGNORECASE,
)

_FINTAB_SCALE_MAP = [
    ('billion', 'e9'),
    ('millions', 'e6'),
    ('million', 'e6'),
    ('thousands', 'e3'),
    ('thousand', 'e3'),
    ('%', 'e-2'),
]


def _fintab_normalize(text: str) -> str:

    s = str(text).strip().lower()
    s = s.replace('$', '').replace('(', '').replace(')', '').replace(',', '')
    s = s.strip()
    return s


def _fintab_unit_free(text: str) -> str:

    s = _fintab_normalize(text)
    s = _FINTAB_UNIT_PATTERN.sub('', s)

    for scale_word, suffix in _FINTAB_SCALE_MAP:
        if scale_word in s:
            s = s.replace(scale_word, '')
            s = s.strip() + suffix
            break

    s = re.sub(r'\s+', '', s)
    return s


def _fintab_float_eq(a: str, b: str) -> bool:

    try:
        fa = float(a)
        fb = float(b)
        return fa == fb
    except (ValueError, TypeError):
        return False


def fintabnetqa_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:

    n_exact = 0
    n_relieved = 0
    n_total = len(predictions)

    for pred_str, ref_str in zip(predictions, references):
        pred_norm = _fintab_normalize(str(pred_str))
        ref_norm = _fintab_normalize(str(ref_str))

        pred_uf = _fintab_unit_free(str(pred_str))
        ref_uf = _fintab_unit_free(str(ref_str))

        is_exact = (pred_norm == ref_norm) or _fintab_float_eq(pred_norm, ref_norm)
        if is_exact:
            n_exact += 1
            n_relieved += 1
            continue

        is_relieved = (
            (pred_uf == ref_uf)
            or _fintab_float_eq(pred_uf, ref_uf)
        )
        if is_relieved:
            n_relieved += 1

    exact_acc = n_exact / max(1, n_total)
    relieved_acc = n_relieved / max(1, n_total)

    return {
        "primary_accuracy": relieved_acc,
        "exact_accuracy": exact_acc,
        "relieved_accuracy": relieved_acc,
        "n_exact": n_exact,
        "n_relieved": n_relieved,
        "n_total": n_total,
    }


DOMAIN_EVAL_FN = {
    "vwtq": wtq_denotation_accuracy,
    "vwtq_syn": wtq_denotation_accuracy,
    "vtabfact": vtabfact_accuracy,
    "fintabnetqa": fintabnetqa_accuracy,
}

DOMAIN_PRIMARY_METRIC_NAME = {
    "vwtq": "Denotation Acc",
    "vwtq_syn": "Denotation Acc",
    "vtabfact": "Classification Acc",
    "fintabnetqa": "Relieved Acc",
}


def evaluate_domain(
    domain: str,
    predictions: List[str],
    references: List[str],
) -> Dict:

    eval_fn = DOMAIN_EVAL_FN.get(domain, None)
    if eval_fn:
        return eval_fn(predictions, references)
    n_correct = sum(
        1 for p, r in zip(predictions, references)
        if _wtq_normalize(str(p)) == _wtq_normalize(str(r))
    )
    acc = n_correct / max(1, len(predictions))
    return {
        "primary_accuracy": acc,
        "exact_match": acc,
        "n_correct": n_correct,
        "n_total": len(predictions),
    }

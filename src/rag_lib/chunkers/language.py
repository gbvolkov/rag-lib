import re
from typing import Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    detect_langs = None  # type: ignore[assignment]

    class LangDetectException(Exception):
        pass

    _HAS_LANGDETECT = False


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_MIN_TEXT_LEN_FOR_LANGDETECT = 20
_MAX_TEXT_LEN_FOR_LANGDETECT = 5000
_MIN_LANGDETECT_CONFIDENCE = 0.55

_ISO_TO_NLTK = {
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "it": "italian",
    "ml": "malayalam",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}


def _detect_script_fallback(text: str, default: str) -> str:
    cyr = len(_CYRILLIC_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    total = cyr + lat
    if total == 0:
        return default
    if cyr > lat and (cyr / total) >= 0.30:
        return "russian"
    if lat >= cyr and (lat / total) >= 0.30:
        return "english"
    return default


def detect_nltk_language(text: str, default: str = "english") -> str:
    """
    Detect NLTK tokenizer language.
    Uses langdetect first (deterministic seed), then falls back to script heuristic.
    """
    if not text or not text.strip():
        return default

    normalized = " ".join(text.split())
    if len(normalized) < _MIN_TEXT_LEN_FOR_LANGDETECT:
        return _detect_script_fallback(normalized, default)

    if _HAS_LANGDETECT and detect_langs is not None:
        try:
            candidates = detect_langs(normalized[:_MAX_TEXT_LEN_FOR_LANGDETECT])
        except LangDetectException:
            candidates = []

        if candidates:
            best = candidates[0]
            mapped = _ISO_TO_NLTK.get(best.lang.lower())
            if mapped:
                if best.prob >= _MIN_LANGDETECT_CONFIDENCE:
                    return mapped
                return _detect_script_fallback(normalized, mapped)

    return _detect_script_fallback(normalized, default)


def resolve_nltk_language(
    language: Optional[str],
    text: str,
    default: str = "english",
) -> str:
    """
    Resolve user-specified language into an NLTK tokenizer language name.
    Supports `auto`, `en`, and `ru` aliases.
    """
    if language is None:
        return detect_nltk_language(text, default=default)

    normalized = language.strip().lower()
    if normalized == "auto":
        return detect_nltk_language(text, default=default)

    if normalized in {"en", "eng", "english"}:
        return "english"
    if normalized in {"ru", "rus", "russian"}:
        return "russian"

    return normalized

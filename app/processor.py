import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTER_RE = re.compile(r"[^a-z0-9\s']")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_keywords(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

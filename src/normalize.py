"""Text normalization for extracted clinical entities and dictionary descriptions."""
from __future__ import annotations

import re

from unidecode import unidecode

_HTML = re.compile(r"<br\s*/?>|&nbsp;|&amp;|<[^>]{1,10}>", re.I)
_HCC = re.compile(r"\((?:hcc|cms[- ]?hcc|cmd|cms|hhs[- ]?hcc|ccc)\)", re.I)
_NONWORD = re.compile(r"[^a-z0-9/+.%<>=\- ]+")
_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    """Light normalization: keeps every token, only fixes encoding/markup/casing/spacing."""
    if not isinstance(text, str):
        return ""
    t = unidecode(text).lower()
    t = _HTML.sub(" ", t)
    t = _HCC.sub(" ", t)
    t = t.replace("_", " ")
    t = _NONWORD.sub(" ", t)
    t = re.sub(r"(?<=[a-z])[.\-/](?=\s|$)", " ", t)  # trailing punctuation on words
    t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)
    return _SPACES.sub(" ", t).strip()


def clean_desc(text: str) -> str:
    """Dictionary descriptions: drop SNOMED semantic tag, then clean."""
    t = re.sub(r"\s*\([^()]*\)\s*$", "", text) if text.endswith(")") else text
    return clean(t)

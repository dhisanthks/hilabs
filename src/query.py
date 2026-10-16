"""Build the set of query variants for an entity: original + abbreviation expansions + spelling repair."""
from __future__ import annotations

from .abbrev import expand
from .normalize import clean

_speller = None


def get_speller():
    global _speller
    if _speller is None:
        from .spell import Speller
        _speller = Speller()
    return _speller


def variants(entity: str, max_variants: int = 6) -> list[str]:
    base = clean(entity)
    sp = get_speller()
    fixed = sp.fix(base)
    out = expand(base)
    if fixed != base:
        out += expand(fixed)
    seen, res = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v); res.append(v)
    return res[:max_variants] or [base]

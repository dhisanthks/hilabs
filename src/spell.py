"""Vocabulary-based spelling repair for OCR / typing noise.

Vocabulary = tokens of every dictionary description (+ training entities). An out-of-vocabulary
token is (a) split into in-vocabulary words if it is two words glued together
("chronickidney"), (b) truncated to its longest in-vocab prefix if it is an OCR stutter
("spondylosispondylosis..."), or (c) replaced by the closest frequent vocabulary word within a
small edit distance ("cardiomyopahty"). Repairs are produced as an extra query variant; the
original stays.
"""
from __future__ import annotations

import collections
import os
import pickle

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .abbrev import ABBREV
from .dicts import CACHE, SYSTEMS, load_system
from .normalize import clean_desc


class Speller:
    def __init__(self, extra_texts=()):
        pkl = os.path.join(CACHE, "vocab.pkl")
        if os.path.exists(pkl):
            vocab = pickle.load(open(pkl, "rb"))
        else:
            vocab = collections.Counter()
            for s in SYSTEMS:
                for t in load_system(s).desc.description.map(clean_desc):
                    vocab.update(t.split())
            pickle.dump(vocab, open(pkl, "wb"))
        self.vocab = collections.Counter(vocab)
        for t in extra_texts:
            self.vocab.update(t.split())
        # candidates for fuzzy replacement: reasonably frequent alphabetic words
        self.words = [w for w, c in self.vocab.items() if c >= 3 and w.isalpha() and len(w) >= 4]
        self.by_len = collections.defaultdict(list)
        for w in self.words:
            self.by_len[len(w)].append(w)
        self.cache = {}

    def known(self, w):
        return self.vocab.get(w, 0) >= 2 or w in ABBREV

    def fix_token(self, w):
        if w in self.cache:
            return self.cache[w]
        out = w
        if len(w) >= 5 and w.isalpha() and not self.known(w):
            out = self._split(w) or self._stutter(w) or self._fuzzy(w) or w
        self.cache[w] = out
        return out

    def _split(self, w):
        best = None
        for i in range(3, len(w) - 2):
            a, b = w[:i], w[i:]
            if self.vocab.get(a, 0) >= 20 and self.vocab.get(b, 0) >= 20:
                sc = min(self.vocab[a], self.vocab[b])
                if best is None or sc > best[0]:
                    best = (sc, f"{a} {b}")
        return best[1] if best else None

    def _stutter(self, w):
        for i in range(len(w) - 1, 5, -1):
            if self.vocab.get(w[:i], 0) >= 5 and len(w) - i >= 4:
                return w[:i]
        return None

    def _fuzzy(self, w):
        maxd = 1 if len(w) <= 6 else 2
        pool = [x for L in range(len(w) - maxd, len(w) + maxd + 1) for x in self.by_len.get(L, ())]
        if not pool:
            return None
        hits = process.extract(w, pool, scorer=Levenshtein.distance, limit=5, score_cutoff=maxd)
        if not hits:
            return None
        # closest first, then most frequent
        hits.sort(key=lambda h: (h[1], -self.vocab[h[0]]))
        return hits[0][0]

    def fix(self, text: str) -> str:
        return " ".join(self.fix_token(t) for t in text.split())

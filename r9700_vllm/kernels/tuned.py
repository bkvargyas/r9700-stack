"""Tuned tile configs (tuning/tune_dense.py -> tuned.json next to this file): {kind: {"N,K": {"M": [WV, SK, NPW]}}}.
A call uses the smallest tuned M bucket >= its M; larger M (prefill) and untuned shapes keep the defaults."""
from __future__ import annotations

import json
import os

_T: dict | None = None


def lookup(kind: str, N: int, K: int, M: int):
    global _T
    if _T is None:
        p = os.environ.get("R9K_TUNED") or os.path.join(os.path.dirname(__file__), "tuned.json")
        try:
            _T = json.load(open(p)) if os.path.exists(p) else {}
        except Exception:
            _T = {}
    e = _T.get(kind, {}).get(f"{N},{K}")
    if not e:
        return None
    ms = sorted(int(m) for m in e)
    b = next((m for m in ms if m >= M), None)
    return tuple(e[str(b)]) if b is not None else None    # beyond the tuned range (prefill): defaults

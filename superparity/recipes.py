"""Recipes = ordered phase schedules (data only). The professor's combinations.

A phase = {n_merges, stage, rule}:
    stage : "sub"   -> --pretokenize whitespace bytelevel  (no merge crosses a space)
            "super" -> --pretokenize bytelevel             (merges may cross a space)
    rule  : "classical" -> --global-merges = n_merges  (global most-frequent pair)
            "parity"    -> --global-merges = 0          (help worst-off language; needs dev)

Four building blocks (the two knobs):
    BPE      = classical + sub      SuperBPE = classical + super
    PA-BPE   = parity    + sub      PA-super = parity    + super

A recipe takes a total merge budget and (for single-transition recipes) a transition
point `t` = how many merges before the schedule switches. The research question
"when should step 2 start?" is exactly a sweep over `t`.
"""

from __future__ import annotations


def _ph(n: int, stage: str, rule: str) -> dict:
    return {"n_merges": n, "stage": stage, "rule": rule}


# --- baselines (no transition) ---------------------------------------------
def classical_bpe(total: int, t: int | None = None) -> list[dict]:
    return [_ph(total, "sub", "classical")]


def pa_bpe(total: int, t: int | None = None) -> list[dict]:
    return [_ph(total, "sub", "parity")]


def pa_super(total: int, t: int | None = None) -> list[dict]:
    """Parity superword from the start (the 4th corner)."""
    return [_ph(total, "super", "parity")]


# --- single-transition recipes (sweep `t`) ---------------------------------
def superbpe(total: int, t: int) -> list[dict]:
    """classical sub -> classical super. (advisor: 80k BPE + 80k SuperBPE)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "super", "classical")]


def bpe_then_pa(total: int, t: int) -> list[dict]:
    """classical sub -> parity sub. (advisor: 80k BPE + 80k PA-BPE)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "sub", "parity")]


def bpe_then_pa_super(total: int, t: int) -> list[dict]:
    """classical sub -> parity super. (parity + superwords combined)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "super", "parity")]


def pa_then_super(total: int, t: int) -> list[dict]:
    """parity sub -> classical super."""
    return [_ph(t, "sub", "parity"), _ph(total - t, "super", "classical")]


# --- three-phase recipe (the one we were missing) --------------------------
def bpe_pa_super(total: int, t: int | None = None) -> list[dict]:
    """classical sub -> parity sub -> classical super. (advisor: BPE -> PA-BPE -> SuperBPE)
    Thirds by default; tune later if a winner."""
    a = total // 3
    b = total // 3
    return [_ph(a, "sub", "classical"), _ph(b, "sub", "parity"),
            _ph(total - a - b, "super", "classical")]


# --- four-phase recipes (advisor's longer schedules) -----------------------
def bpe_super_pa(total: int, t: int | None = None) -> list[dict]:
    """40k BPE + 40k SuperBPE + 80k PA-super -> 1/4 sub, 1/4 super, 1/2 parity-super."""
    a, b = total // 4, total // 4
    return [_ph(a, "sub", "classical"), _ph(b, "super", "classical"),
            _ph(total - a - b, "super", "parity")]


def bpe_pa_super_pa(total: int, t: int | None = None) -> list[dict]:
    """40k BPE + 40k PA-BPE + 40k SuperBPE + 40k PA-super -> four equal quarters."""
    q = total // 4
    return [_ph(q, "sub", "classical"), _ph(q, "sub", "parity"),
            _ph(q, "super", "classical"), _ph(total - 3 * q, "super", "parity")]


# Registry: name -> (builder, needs_transition)
RECIPES = {
    "classical_bpe":     (classical_bpe,     False),
    "pa_bpe":            (pa_bpe,            False),
    "pa_super":          (pa_super,          False),
    "superbpe":          (superbpe,          True),
    "bpe_then_pa":       (bpe_then_pa,       True),
    "bpe_then_pa_super": (bpe_then_pa_super, True),
    "pa_then_super":     (pa_then_super,     True),
    "bpe_pa_super":      (bpe_pa_super,      False),
    "bpe_super_pa":      (bpe_super_pa,      False),
    "bpe_pa_super_pa":   (bpe_pa_super_pa,   False),
}


def needs_dev(phases: list[dict]) -> bool:
    return any(ph["rule"] == "parity" for ph in phases)


def final_stage(phases: list[dict]) -> str:
    """Stage of the last phase -> which pretokenizer to evaluate the tokenizer with."""
    return phases[-1]["stage"]


def build(name: str, total: int, transition: int | None = None) -> list[dict]:
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; choices: {sorted(RECIPES)}")
    builder, needs_t = RECIPES[name]
    if needs_t and transition is None:
        raise ValueError(f"recipe {name!r} requires a transition point")
    return builder(total, transition)

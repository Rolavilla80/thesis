"""RQ2 recipes -- the advisor's combination options as phase schedules (data only).

A recipe is an ordered list of phases; each phase is {n_merges, stage, merge_rule}:
    stage      : "sub"  (no merge crosses a space) | "super" (merges may cross a space)
    merge_rule : "classical" (global most-frequent) | "parity" (help worst-off language)

The four building blocks (knob combinations):
    BPE        = classical + sub      SuperBPE = classical + super
    PA-BPE     = parity    + sub      PA-super = parity    + super

Recipes are built from a total merge budget and (for single-transition recipes) a
transition point `t` = how many merges happen before the schedule switches. The
research question "when should step 2 start?" is exactly a sweep over `t`.

Parity phases need a multi-parallel dev set passed at run time (devs=...); classical
phases ignore it. Vocab size ~= base byte-level alphabet (256) + total_merges, so for
a 20k vocab use total_merges ~= 19_744 (see DEFAULT_TOTAL_MERGES).
"""

from __future__ import annotations

# Locked decisions (see notes/START_HERE.md).
LANGUAGES = ["eng_Latn", "deu_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans",
             "jpn_Jpan", "hin_Deva", "tha_Thai", "kor_Hang", "tur_Latn"]
DEFAULT_VOCAB = 20_000
DEFAULT_TOTAL_MERGES = DEFAULT_VOCAB - 256  # byte-level alphabet is 256 tokens


def _ph(n: int, stage: str, rule: str) -> dict:
    return {"n_merges": n, "stage": stage, "merge_rule": rule}


# --- baselines (no transition) ---------------------------------------------
def classical_bpe(total: int, t: int | None = None) -> list[dict]:
    """Plain BPE: classical subword only."""
    return [_ph(total, "sub", "classical")]


def pa_bpe(total: int, t: int | None = None) -> list[dict]:
    """Parity-aware BPE: parity subword only (needs dev set)."""
    return [_ph(total, "sub", "parity")]


# --- single-transition recipes (sweep `t`) ---------------------------------
def superbpe(total: int, t: int) -> list[dict]:
    """SuperBPE: classical subword, then classical superword. (advisor: 80k BPE + 80k Super)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "super", "classical")]


def bpe_then_pa(total: int, t: int) -> list[dict]:
    """Classical subword warm-up, then parity subword. (advisor: 80k BPE + 80k PA-BPE)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "sub", "parity")]


def bpe_then_pa_super(total: int, t: int) -> list[dict]:
    """Classical subword, then parity *superword*. (advisor: BPE then PA + SuperBPE together)"""
    return [_ph(t, "sub", "classical"), _ph(total - t, "super", "parity")]


def pa_then_super(total: int, t: int) -> list[dict]:
    """Parity subword, then classical superword."""
    return [_ph(t, "sub", "parity"), _ph(total - t, "super", "classical")]


# --- multi-phase recipes (fixed fractions; advisor's longer schedules) ------
def bpe_super_pa(total: int, t: int | None = None) -> list[dict]:
    """40k BPE + 40k SuperBPE + 80k PA-BPE -> here 1/4 sub, 1/4 super, 1/2 parity-super."""
    a = total // 4
    b = total // 4
    c = total - a - b
    return [_ph(a, "sub", "classical"), _ph(b, "super", "classical"),
            _ph(c, "super", "parity")]


def bpe_pa_super_pa(total: int, t: int | None = None) -> list[dict]:
    """40k BPE + 40k PA-BPE + 40k SuperBPE + 40k PA-BPE -> four equal quarters."""
    q = total // 4
    return [_ph(q, "sub", "classical"), _ph(q, "sub", "parity"),
            _ph(q, "super", "classical"), _ph(total - 3 * q, "super", "parity")]


# Registry: name -> (builder, needs_transition)
RECIPES = {
    "classical_bpe":     (classical_bpe,     False),
    "pa_bpe":            (pa_bpe,            False),
    "superbpe":          (superbpe,          True),
    "bpe_then_pa":       (bpe_then_pa,       True),
    "bpe_then_pa_super": (bpe_then_pa_super, True),
    "pa_then_super":     (pa_then_super,     True),
    "bpe_super_pa":      (bpe_super_pa,      False),
    "bpe_pa_super_pa":   (bpe_pa_super_pa,   False),
}


def needs_dev(phases: list[dict]) -> bool:
    """True if any phase uses the parity rule (and therefore needs a dev set)."""
    return any(ph["merge_rule"] == "parity" for ph in phases)


def build_recipe(name: str, total_merges: int = DEFAULT_TOTAL_MERGES,
                 transition: int | None = None) -> list[dict]:
    """Materialize a named recipe into a list of phases."""
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; choices: {sorted(RECIPES)}")
    builder, needs_t = RECIPES[name]
    if needs_t and transition is None:
        raise ValueError(f"recipe {name!r} requires a transition point")
    return builder(total_merges, transition)

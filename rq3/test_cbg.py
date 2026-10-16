# Unit tests for the _cbg patch (DESIGN note §3/§5.7). Run from the repo root:
#   WSL:  uv run python test_cbg.py
# Prints one line per case; exits nonzero on any failure.

import sys, os, types, multiprocessing

_orig = multiprocessing.get_context
multiprocessing.get_context = lambda m=None: _orig("spawn") if m == "forkserver" else _orig(m)
def stub(name, path=None, **attrs):
    m = types.ModuleType(name)
    if path: m.__path__ = [path]
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m
root = os.path.abspath("script_bpe")
stub("script_bpe", root)
from script_bpe.pretokenize import get_pretokenizer  # noqa: E402

ZWJ = "‍"
cbg = get_pretokenizer("bytes_gpt4o_cbg")
nov = get_pretokenizer("bytes_gpt4o_cbg_novirama")
scbg = get_pretokenizer("scriptenc_gpt4o_cbg")

# (label, text, expected violates_grapheme_policy under cbg)
CASES = [
    # L1: starts with a continuation mark (incl. after leading space)
    ("L1 Thai mark-start 'ิน'", "ิน", True),
    ("L1 space+mark ' ัน' (loophole covered)", " ัน", True),
    ("L1 Devanagari 'ार' (SpacingMark)", "ार", True),
    ("L1 Hebrew 'ָה' (qamats+he)", "ָה", True),
    ("L1 Arabic 'ًا' (fathatan+alef)", "ًا", True),
    ("L1 starts with virama '्र'", "्र", True),
    ("L1 starts with ZWJ", ZWJ + "👩", True),
    ("L1 skin-tone modifier start '🏽x'", "🏽x", True),
    # L2 / R4: Hangul jamo edges
    ("L2 starts with V jamo 'ᅡ요'", "ᅡ요", True),
    ("R4 ends with L jamo '가ᄒ'", "가ᄒ", True),
    # R1/R3
    ("R1 ends with ZWJ", "👨" + ZWJ, True),
    ("R3 ends with Prepend 'a\\u0600'", "a؀", True),
    # R5: flags are RI pairs
    ("R5 odd RI 'a🇹'", "a🇹", True),
    ("R5 whole flag '🇹🇭' ok", "🇹🇭", False),
    ("R5 three RI '🇹🇭🇩'", "🇹🇭🇩", True),
    # allowed: complete clusters, single chars, right-edge marks (the ceiling)
    ("whole word 'กิน' ok", "กิน", False),
    ("ends with mark 'ก่' ok (complete cluster; ceiling case)", "ก่", False),
    ("emoji+modifier '👍🏽' ok", "👍🏽", False),
    ("plain latin 'cat' ok", "cat", False),
    ("single mark 'ั' ok (atomic assembly)", "ั", False),
    ("single ZWJ ok (atomic assembly)", ZWJ, False),
    ("spaces only ok", "  ", False),
]

fails = 0
for label, text, want in CASES:
    got = cbg.violates_grapheme_policy(text)
    ok = got == want
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: violates={got} (expected {want})")

# R2 toggle: virama-final blocked by default, allowed under _novirama
for label, text in [("R2 'क्' (conjunct cut)", "क्"), ("R2 word-final virama 'जगत्' (documented cost)", "जगत्")]:
    a, b = cbg.violates_grapheme_policy(text), nov.violates_grapheme_policy(text)
    ok = a is True and b is False
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: cbg={a} novirama={b} (expected True/False)")

# integration: the rule fires through token_allowed on BOTH encodings
for pt, name in [(cbg, "bytes_cbg"), (scbg, "script_cbg")]:
    seq_bad = [t for e in pt.encode_text("ิน") for t in e.atomic_token_ids]
    seq_ok = [t for e in pt.encode_text("กิน") for t in e.atomic_token_ids]
    ok = (not pt.token_allowed(seq_bad)) and pt.token_allowed(seq_ok)
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  token_allowed wiring on {name}: 'ิน' blocked, 'กิน' allowed")

# regression: hashes of pre-existing configs must be unchanged (corpus caches stay valid)
h = get_pretokenizer("bytes_gpt4o_cb").hash()
ok = h == "PT-ca6443c4"
fails += not ok
print(f"  {'PASS' if ok else 'FAIL'}  bytes_gpt4o_cb hash unchanged: {h} (expected PT-ca6443c4)")
print(f"  new cbg hashes: bytes={cbg.hash()}  script={scbg.hash()}  novirama={nov.hash()}")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)

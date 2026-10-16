import sys, os, types, multiprocessing, unicodedata
# --- import only the pretokenize subpackage (skip the heavy package __init__), and fake forkserver on Windows
_orig = multiprocessing.get_context
multiprocessing.get_context = lambda m=None: _orig("spawn") if m == "forkserver" else _orig(m)
pkg = types.ModuleType("script_bpe"); pkg.__path__ = [os.path.abspath("script_bpe")]; sys.modules["script_bpe"] = pkg
from script_bpe.pretokenize import get_pretokenizer

def show(title, pt, cases):
    print(f"\n### {title}")
    for label, a, b in cases:
        ok = pt.bpe_merge_allowed(a, b)
        print(f"  {'ALLOWED ' if ok else 'BLOCKED '}  {label}")

# ---------- BYTES ----------
cb  = get_pretokenizer("bytes_gpt4o_cb")
cbi = get_pretokenizer("bytes_gpt4o_cbi")
B = lambda s: [cb.byte_ids[x] for x in s.encode("utf-8")]     # helper: text -> list of byte token ids
raw = lambda bs: [cb.byte_ids[x] for x in bs]                    # helper: raw bytes -> ids

thai_ko = "ก"   # E0 B8 81
print("UTF-8 of ก:", thai_ko.encode("utf-8").hex(" "))
show("bytes_gpt4o_cb  (paper's constraint; RQ1 cases)", cb, [
    ("case: lead+cont  E0 + B8            (partial prefix, left-to-right)", raw(b"\xe0"), raw(b"\xb8")),
    ("case: prefix+cont  E0B8 + 81        (completes ก)",                  raw(b"\xe0\xb8"), raw(b"\x81")),
    ("case 3: full+partial  ' ' + E0      (space + lead byte)",             B(" "), raw(b"\xe0")),
    ("case 3: full+partial  ก + E0B8      (char + 2-byte prefix)",          B("ก"), raw(b"\xe0\xb8")),
    ("case 4: cont+cont  B8 + 81",                                          raw(b"\xb8"), raw(b"\x81")),
    ("case 5: partial+full  E0B8 + ' '",                                     raw(b"\xe0\xb8"), B(" ")),
    ("full+full  ก + ก",                                                    B("ก"), B("ก")),
    ("full+full  'ca' + 'fé'",                                              B("ca"), B("fé")),
])

# what survives as 'undecodable' under cb -> exactly what Table 2's column counts
print("\n### Table-2 column = tokens whose bytes do NOT strict-decode. Under _cb the ONLY such tokens allowed:")
for bs in [b"\xe0", b"\xe0\xb8", b"\xf0\x9f\x98", b"\xe0\xb8\x81", b"\x20\xe0"]:
    seq = raw(bs); dec = cb.try_decode_strict(seq)
    print(f"  {bs.hex(' '):12}  decodes={str(dec is not None):5}  allowed_as_token={cb.token_allowed(seq)}  "
          f"{'<- pure single-char prefix, counted in the 209/183' if dec is None and cb.token_allowed(seq) else ''}")

# ---------- enforce_inherited (cbi) ----------
acute = "\u0301"   # COMBINING ACUTE ACCENT, Script=Inherited
print("\n### enforce_inherited: 'inherited_chars' size =", len(cbi.inherited_chars),
      "| U+0301 in set:", acute in cbi.inherited_chars,
      "| Thai SARA I U+0E34 in set:", "\u0e34" in cbi.inherited_chars,
      "| Devanagari virama U+094D in set:", "\u094d" in cbi.inherited_chars,
      "| ZWJ U+200D in set:", "\u200d" in cbi.inherited_chars)
cases = [
    ("e + ◌́        (assembles a grapheme)",            B("e"), B(acute)),
    ("◌́ + s        (fragment glued to foreign text)",   B(acute), B("s")),
    ("caf + é(NFD: e+◌́)  whole graphemes both sides", B("caf"), B("e"+acute)),
    ("cafe + ◌́     (RIGHT-cut: content can't tell)",   B("cafe"), B(acute)),
    ("ก + ◌ิ  Thai vowel sign (Script=Thai, not Inherited)", B("ก"), B("\u0e34")),
    ("◌ิ + ก  Thai vowel sign FIRST",                    B("\u0e34"), B("ก")),
]
show("bytes_gpt4o_cb  (no inherited rule)", cb, cases)
show("bytes_gpt4o_cbi (enforce_inherited=True)", cbi, cases)

# ---------- NFC ----------
print("\n### normalization default:", cb.config.normalization)
nfd = "cafe\u0301"
print(f"  input NFD 'cafe\u0301' len={len(nfd)}  ->  normalize() -> {cb.normalize(nfd)!r} len={len(cb.normalize(nfd))}  (é became 1 codepoint)")
print(f"  Thai 'กิ' len={len('กิ')} -> after NFC len={len(cb.normalize('กิ'))}  (stays 2 codepoints: no precomposed form)")
print(f"  Hindi 'क्ष' len={len('क्ष')} -> after NFC len={len(cb.normalize('क्ष'))}  (conjunct stays 3 codepoints)")

# ---------- SCRIPT ----------
sc = get_pretokenizer("scriptenc_gpt4o_cb")
enc = lambda s: [tid for ch in sc.encode_text(s) for tid in ch.atomic_token_ids]
print("\n### SCRIPT encoding of 'กa':", [sc.atomic_tokens[t] for t in enc("กa")])
blk, idx = enc("ก")
blk2, idx2 = enc("a")
show("scriptenc_gpt4o_cb", sc, [
    ("block + index  (completes ก)", [blk], [idx]),
    ("index + block  (crosses characters)", [idx], [blk2]),
    ("ก + block(a)   (full + partial)", [blk, idx], [blk2]),
    ("ก + a          (full + full)", [blk, idx], [blk2, idx2]),
])

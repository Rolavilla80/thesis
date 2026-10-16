r"""Bridge our learned tokenizers into the swiss-ai TokEval suite
(github.com/swiss-ai/tokenizer-intrinsic-evals).

For each recipe we build the HF BPE tokenizer with the SAME pretokenizer metrics.py uses
(sub = Whitespace+ByteLevel, super = ByteLevel no-split), attach a ByteLevel decoder, and
save it as tokenizer.json. We register it with class "hf" so TokEval loads it via
Tokenizer.from_file() and KEEPS our pretokenizer -> superwords stay enabled at eval time.
(TokEval's "custom_bpe" class hardcodes a Whitespace pretokenizer, which would silently
disable every superword -> do NOT use it for SuperBPE tokenizers.)

Outputs into ./tokeval_export/:
  <recipe>/tokenizer.json   one per recipe
  tokenizers.json           TokEval tokenizer config (class: hf, absolute paths)
  languages.json            our 10 langs -> FLORES devtest (parallel), with script groups

Then, from a clone of the TokEval repo (uv synced):
  uv run tokenizer-analysis \
    --tokenizer-config  <ABS>/tokeval_export/tokenizers.json \
    --language-config   <ABS>/tokeval_export/languages.json \
    --measurement-config configs/text_measurement_config_lines.json
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from run import read_merges
from metrics import build_hf_tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# label, single-run folder name, final stage
ITEMS = [
    ("classical_bpe", "classical_bpe_v80000",   "sub"),
    ("pa_bpe",        "pa_bpe_v80000",          "sub"),
    ("superbpe",      "superbpe_v80000_t60000", "super"),
    ("bpe_pa_super",  "bpe_pa_super_v80000",    "super"),
    ("bpe_super_pa",  "bpe_super_pa_v80000",    "super"),
]

NAMES = {"eng_Latn": "English", "deu_Latn": "German", "rus_Cyrl": "Russian",
         "arb_Arab": "Arabic", "zho_Hans": "Chinese", "jpn_Jpan": "Japanese",
         "hin_Deva": "Hindi", "tha_Thai": "Thai", "kor_Hang": "Korean", "tur_Latn": "Turkish"}
ISO = {"eng_Latn": "en", "deu_Latn": "de", "rus_Cyrl": "ru", "arb_Arab": "ar",
       "zho_Hans": "zh", "jpn_Jpan": "ja", "hin_Deva": "hi", "tha_Thai": "th",
       "kor_Hang": "ko", "tur_Latn": "tr"}
SCRIPT_FAMILY = {"Latin": ["eng_Latn", "deu_Latn", "tur_Latn"], "Cyrillic": ["rus_Cyrl"],
                 "Arabic": ["arb_Arab"], "Han": ["zho_Hans"], "Japanese": ["jpn_Jpan"],
                 "Devanagari": ["hin_Deva"], "Thai": ["tha_Thai"], "Korean": ["kor_Hang"]}


def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def main():
    out = config.HERE / "tokeval_export"
    out.mkdir(parents=True, exist_ok=True)

    tok_cfg = {"plot_tokenizers": [label for label, _, _ in ITEMS]}
    for label, folder, stage in ITEMS:
        mp = config.RESULTS / "single" / folder / f"{folder}.merges"
        if not mp.exists():
            print(f"  MISSING: {mp}  (skipping)")
            continue
        tk = build_hf_tokenizer(read_merges(mp), stage=stage)
        tk.decoder = ByteLevelDecoder()
        tdir = out / label
        tdir.mkdir(exist_ok=True)
        tjson = tdir / "tokenizer.json"
        tk.save(str(tjson))
        tok_cfg[label] = {"class": "hf", "path": _posix(tjson)}
        print(f"  {label:14s} -> tokenizer.json  (vocab {tk.get_vocab_size()}, stage={stage})")

    (out / "tokenizers.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")

    langs = {code: {"name": NAMES[code], "iso_code": ISO[code],
                    "data_path": _posix(config.FLORES / f"{code}.devtest.txt")}
             for code in config.LANGUAGES}
    lang_cfg = {"languages": langs, "analysis_groups": {"script_family": SCRIPT_FAMILY}}
    (out / "languages.json").write_text(json.dumps(lang_cfg, indent=2), encoding="utf-8")

    print(f"\n-> {out / 'tokenizers.json'}\n-> {out / 'languages.json'}")
    print("Run TokEval (from its repo clone) with these two configs (see this file's docstring).")


if __name__ == "__main__":
    main()

"""RQ2 tokenizer engine.

`parity_aware_bpe` is a FROZEN, byte-for-byte copy of
`parity_aware_bpe/parity_aware_learn_bpe_original.py` (Sennrich & Foroutan's
parity-aware BPE learner). It is copied here so the RQ2 work is self-contained
and immune to any future edits to the top-level RQ1 parity files.

Why the original (and not the top-level parity_aware_learn_bpe.py)?
The top-level file is `original + RQ1's merge_check_strategy subsystem`, which is
fully gated behind non-'none' strategies. Under merge_check_strategy='none' (all
RQ2 ever uses) it is algorithmically identical to the original, so RQ2 carries the
clean original and none of the RQ1 code rides along.

`phased_runner` drives this engine: it flips two knobs per phase --
  parity vs classical  (num_global)            -- which pair to merge
  subword vs superword (pretokenizer regex)    -- may a merge cross a space
-- and chains phases by preloading the previous phase's cumulative merges. This is
the one-engine-with-a-schedule that realizes every recipe in the RQ2 design.
"""

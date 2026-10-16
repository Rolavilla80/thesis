# ⚠️ LEGACY — this folder is superseded

This is the **retired June 2026 build** of the RQ2 experiments. It is kept for
history (and its `data/` folder is still referenced as the corpus source), but
**its code and notes do not describe the current pipeline or results.**

Known differences from the live build (do not review this folder for the thesis):

- It injects the SuperBPE **paper's STAGE1/STAGE2 regex**; the live build uses the
  curated engine's native pretokenizers instead (advisor direction).
- Its three-phase recipes existed as design only; the live build has trained them
  at four vocabulary scales — they are the headline results.

**The live build is [`../superparity/`](../superparity/).
Start at [`../superparity/notes/OVERVIEW.md`](../superparity/notes/OVERVIEW.md).**

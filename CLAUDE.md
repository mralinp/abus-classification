# abus-classification: working rules

## One notebook per reproduced paper

Every paper we reproduce (roadmap Phase 2 in `README.md`) gets its own notebook in `notebooks/`, numbered in the
order we do them (`0-visualization`, `1-tao2013`, `2-barekatrezaei2026`, `3-zhou2021`, ...). The code lives in a
`reproduce_<author><year>.py` script at the repo root; the notebook imports it, it does not duplicate it.

Each notebook must contain, so a reader can check the implementation is correct:

1. **Paper summary and reference**: what the method is, the citation, and which sections/equations we implement.
2. **Implementation details**: every choice the paper specifies (with section/equation numbers) and every
   **deviation or assumption** where the paper is silent or our data differs, in a table (paper vs ours).
3. **Verification of the implementation**: checks that the code does what the paper says, e.g. tensor shapes and
   parameter counts, a hand-computed value for a loss or feature, an overfit-one-batch sanity run, a unit test
   reference. Not only the final metric.
4. **Images**: figures from the paper's method (architecture or pipeline diagram redrawn or described), our
   intermediate outputs (feature maps, segmentation overlays, ROC curves, training curves).
5. **Tables**: paper's reported numbers next to ours (same metrics), the difference, seed variance, and the
   paper's ablations where we ran them.
6. **Protocol**: data split, positive class, number of seeds, how to re-run (including the Kaggle kernel in
   `kaggle/`), and where the stored results are (`results/<paper>/`).
7. **Open issues**: what does not reproduce and the current hypotheses.

Notebooks load stored results from `results/`; they must not need a multi-hour run to open.

## Conventions

- Protocol for all methods: train on Train+Validation (130 volumes), report on the official Test split (70),
  malignant = positive class, AUC with bootstrap CI, 3 seeds.
- Heavy training runs on Kaggle GPU kernels (`kaggle/<name>/`, private datasets `simayousef/tdsc-abus2023-*`);
  the feature cache `data/eswa2025.npz` is the output of the `eswa2025-reproduction` kernel.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Write the user's name as "Ali Naderiparizi" in citations.

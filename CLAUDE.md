# abus-classification: working rules

## Where code goes

This repo is the library for every ABUS classification paper we reproduce, plus utilities for researchers.
Everything reusable lives in the `abus_classification` package, never in root scripts or notebooks:

- `papers/<author><year>/`: one subpackage per paper (`model.py`, `losses.py`, `train.py`/`pipeline.py`,
  `__init__.py` exporting `main`, `__main__.py` CLI): `python -m abus_classification.papers.<name>`.
- `protocol.py`: the shared evaluation protocol (device, metrics, folds, bootstrap CI).
- `challenge.py`: the TDSC-ABUS 2023 challenge evaluation and published leaderboard, reported by every paper.
- `data/`: dataset-level caches (`lesion_cache.py`: patches, radiomic and Laplace-Beltrami features per volume).
- `features/`, `models/`, `transformers/`, `preprocessing/`, `utils/`: reusable building blocks. If a second
  paper needs something a paper subpackage defined, move it to one of these.
- `test/papers_tests/`: shape, hand-computed-value and metric tests for each paper.
- Root `reproduce_*.py` files are only back-compat shims for the Kaggle kernels; do not add new ones.

## One branch per paper

Implement papers one by one, each on its own branch (`feat/<author><year>`) with its own PR; never put two papers'
work on one branch. Compare methods afterwards, in a separate comparison step (roadmap: unified comparison table).
Shared code (protocol, data cache, utilities) goes in its own PR/branch first; a paper branch may build on it.
Kaggle kernels clone the paper's own branch, so keep `kaggle/<name>/run.py` pointing at it.

## One notebook per reproduced paper

Every paper we reproduce (roadmap Phase 2 in `README.md`) gets its own notebook in `notebooks/`, numbered in the
order we do them (`0-visualization`, `1-tao2013`, `2-barekatrezaei2026`, `3-zhou2021`, ...). The code lives in
`abus_classification/papers/<author><year>/`; the notebook imports it, it does not duplicate it.

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
8. **Challenge benchmark**: the TDSC-ABUS 2023 evaluation (`abus_classification/challenge.py`) on stored predictions,
   placed on the published leaderboard for every task the method does, stating whether it located the lesion itself or
   used the ground truth. See `notebooks/challenge-benchmark.ipynb`.

Notebooks load stored results from `results/`; they must not need a multi-hour run to open.

## Conventions

- Protocol for all methods: train on Train+Validation (130 volumes), report on the official Test split (70),
  malignant = positive class, AUC with bootstrap CI, 3 seeds.
- Kaggle runs save what the benchmark needs: test probabilities and, if the method produces them, full-volume masks and
  detection boxes.
- Heavy training runs on Kaggle GPU kernels (`kaggle/<name>/`, private datasets `tdsc-abus2023-*`, account `mralinp`, pushed with `python kaggle/push.py <name>`);
  the feature cache `data/eswa2025.npz` is the output of the `eswa2025-reproduction` kernel.
- Write the user's name as "Ali Naderiparizi" in citations.
- Only name `mralinp` as the Kaggle account. Committed kernel metadata lists sources without an owner; other owners go in
  `~/.kaggle/owners.json`, never in the repo.

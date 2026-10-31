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

## One document per paper and job

Every paper gets `docs/reproductions/<author><year>.md`, and every other job (a benchmark, a comparison, a shared cache)
gets `docs/reproductions/<name>.md`. **A PR that adds or changes a paper or job, its code, its Kaggle kernel or its stored
results includes the document, new or updated in that same PR.** A PR without it is not ready to merge.

The notebook shows the evidence; the document tells someone how to rerun the work and get the same results. Link to the
notebook instead of copying its tables. Each document contains:

1. **Implementation**: what was built and where (modules, entry points, tests), how the data flows through it, and the
   decisions and deviations that matter, with a link to the notebook's table for the full list.
2. **Requirements**: data and caches needed, where they come from, and the hardware.
3. **Reproducing the results**: the exact commands, in order, from a clean checkout to the stored results, including every
   Kaggle kernel: its name, what it reads and writes, the branch it clones, the settings used, and how long it took.
4. **Expected results**: the numbers a rerun should give, with seed variation, and where they are stored (`results/<name>/`).
5. **Status**: whether the stored results come from the current code, what has not been run, and any commit, parameter or
   environment that differs from the instructions.

Only state commands and run times that were actually used. Write "not run yet" for anything that was not.

## How we run a paper

Do these in order, for every paper:

1. Mark the paper 🔄 in the `README.md` roadmap.
2. Implement it on its own branch with its tests, Kaggle kernel, notebook and document. Test it on real volumes locally before using GPU quota.
3. Open the PR. Keep its text to what the document says, and update both as results arrive.
4. Push the kernel and wait until it finishes, then store the results and update the notebook, document and README.
5. Write every open question, unexplained result or deferred piece of work as a GitHub issue with a fitting label (`question`, `bug`, `enhancement`, `documentation`), so nothing depends on memory.

Every Kaggle training run must:

- **Save to Drive.** Sync checkpoints, final trained weights, logs and results to `abus-checkpoints/<name>` with `utils.drive.DriveSync`, which needs the `RCLONE_CONF` secret and so a start from Kaggle's editor. Stop the editor session afterwards, or a second copy trains.
- **Resume.** Save a checkpoint every epoch, skip finished runs, and stop before the 12 h session limit so the output is kept.
- **Use all the hardware.** Check for T4 x 2 at the start, run one training per GPU or use both in one model, and parallelise CPU work across all cores.
- **Log properly.** Write timestamped lines for each epoch to stdout and a progress file: loss, learning rate, validation metric, best so far, time.
- **Stop early.** Use an LR scheduler and early stopping on a volume-level validation set; record it as a deviation if the paper differs.

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

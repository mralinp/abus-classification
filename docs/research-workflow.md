# Research workflow

This project collects reusable methods and evidence for medical image analysis in 3D automated breast
ultrasound. Its current task is to select relevant papers, implement their methods, evaluate them on
TDSC-ABUS 2023 and report what can be established. The [research index](papers.md) connects each study
to its artifacts. Phase 3 will investigate Laplace–Beltrami shape representations and their
classification performance against the reproduced baselines.

## Select a study

Record the citation, task, method, available code/data, why the paper was selected and which claims
will be examined. Relevance to ABUS, influence, reported performance, methodological interest and
coverage of competing approaches can motivate selection. Selection is not an endorsement. Distinguish
a paper's original experiments from applying its method to TDSC-ABUS; a change of dataset or protocol
limits conclusions about reproducing its reported score.

## Keep one connected record per paper

| Artifact | Responsibility |
| --- | --- |
| `abus_classification/papers/<author><year>/` | Reusable method implementation and entry points; shared components belong in the library |
| `notebooks/` | Paper explanation, implementation verification, figures, original and reproduced results, limitations and open questions |
| `kaggle/<study>/` | GPU training, evaluation and diagnostic run scripts; multiple kernels may belong to one study |
| `results/<study>/` | Stored predictions, metrics and histories needed to audit the reported experiments |
| `docs/reproductions/<author><year>.md` | Detailed account of what was done, how, why, how to repeat it, and what remains uncertain |

Notebooks should load expensive experiment outputs rather than start a long training job when opened.
Identify any data-dependent figures or checks separately. Classical methods may live in a shared feature
module and use notebook-based fitting; record that exception rather than claim a missing kernel exists.
Historical thesis notebooks remain available, but are not current benchmark evidence without a
recorded dataset, protocol and implementation check.

## Implement and verify

Map paper sections, equations and diagrams to code. In the notebook and reproduction record, separate
paper-specified settings, unspecified details we chose, known deviations, and details awaiting author
clarification. A reconstruction from paper text is not the authors' original implementation.

Use checks that address the method: tensor shapes, parameter counts, hand-computed features or losses,
synthetic examples and a small training sanity check where appropriate. Unit tests establish specific
properties; they do not establish fidelity to every paper detail or validate a clinical claim.

## Record protocols precisely

The shared target is fitting Train + Validation (130 volumes) and reporting on the official Test split
(70), with malignant as positive, three seeds, AUC with confidence intervals, and accuracy. Record the
probability threshold, seed aggregation and uncertainty calculation. Distinguish mean-of-seed metrics
from metrics of an ensemble of seed probabilities.

Report the original paper's protocol separately when implemented. A variant that holds out the official
Validation split for early stopping fits Train only (100), so its results cannot be presented as the
same training procedure as fitting all 130 volumes. Keep a volume's slices and patches together in
splits and folds. Fit preprocessing within training folds when required; document existing departures
and their implications. Keep Test out of model selection and label exploratory Test diagnostics.

State whether inputs use ground-truth masks, official boxes, predicted segmentations or detections.
Use `abus_classification/challenge.py` for supported challenge metrics and explain its approximations.
A classification-only method has no overall three-task challenge rank. Ground-truth-assisted scores
are contextual comparisons with challenge teams, not equivalent end-to-end submissions. Predicted-mask
evaluation and pairwise statistical comparisons are plans until backed by recorded runs.

## Record behavior, interventions and outcomes

A reproduction guide must let someone follow the reasoning as well as repeat the commands. For every
unexpected behavior or intervention, record:

1. **Observation:** what happened, with run ID, history/figure and affected metric. Separate evidence
   of a train/validation gap from a proposed explanation such as memorization or leakage.
2. **Paper requirement and our choice:** cite the source or mark the detail unspecified. Identify the
   original setting, the changed setting and the reason for changing it.
3. **Experiment:** exact code revision, command/kernel version, data split, seeds, preprocessing,
   dependencies, hardware, stopping rule and output locations. Mark missing historical provenance unknown.
4. **Outcome:** original and changed results, uncertainty where available, and whether the experiment
   supports the proposed explanation. Preserve unsuccessful interventions as well as successful ones.
5. **Scope:** identify the variant explicitly and list what is still untested. A lower score or an
   unsuccessful intervention alone does not establish that the original paper is wrong.

For example, the [Yu 2024 record](reproductions/yu2024.md) documents rapid training memorization,
introducing early stopping and an LR scheduler, fitting fewer volumes, and the mixed resulting scores.
Those changes belong to a diagnostic variant, alongside the paper-default runs.

## Make GPU runs auditable

Kaggle is the current GPU training platform; see the [kernel guide](../kaggle/README.md). Record the
source commit and kernel version rather than only a moving branch, along with accelerator, dependency
versions, input data/cache provenance, seeds, settings, runtime and collected artifacts. A future run
on another GPU server needs the same record. Keep credentials and private dataset contents out of Git.

For each documented command, distinguish a historical command known to have run, an invocation derived
from the current source, and a command not yet tested. Do not invent timings, exact revisions or
artifact recovery steps. Record missing checkpoints or predictions when aggregate tables alone survive.

## Invite original-author verification

Prepare a review package with the paper citation and targeted claims, an exact implementation revision,
the notebook and reproduction guide, a paper-versus-implementation table, verification checks, protocol,
results and specific questions. Ask authors to clarify unresolved choices, review fidelity and contribute
source or corrections. Record review scope: an architecture review is not verification of every run.

Track source availability and author review separately from implementation and experiment status:
awaiting source, review not yet requested, review requested, feedback received, corrections pending,
or verified for a stated revision and scope. Use a status only when supported by the record. Preserve
feedback-driven changes and identify which experiments need rerunning. Do not publish private
correspondence or contributed code without the necessary permission/license. Sending outreach is a
separate action requiring an explicit instruction; this documentation change sends no messages.

For Barekatrezaei 2026, finalized source including the CNN code is awaited from the first author.
Existing runs remain provisional and fully recorded in its notebook and reproduction guide. README
comparisons are deferred until source comparison and reruns support a reviewed assessment.

## Summarize and integrate

The README summarizes the current direction and supported results. Detailed findings, including
unresolved differences and failed attempts, remain in the study notebook and reproduction record.
README omission must not erase or relabel a provisional experiment as a success. Make conclusions
specific to the implementation, data and protocol, and explain what another researcher can rely on.

### Landing-page review for every PR

The project homepage is rendered from `README.md` through `index.md`. Review it alongside each PR's
research changes: purpose and hero metadata, study status, supported result summaries, roadmap, linked
records and relevant images should agree. Use figures that explain the current results and give them
accurate captions, descriptive alt text and traceable sources. Preserve the distinction between
provisional findings, completed experiments and author verification.

Check navigation, figures, wide tables and links on the deployed base path as well as GitHub. Notebooks
and internal guides excluded by Jekyll should link to GitHub, not missing pages on the website. For
presentation changes, build with the locked Jekyll dependencies and inspect desktop/mobile and
light/dark rendering when available. The PR should record the landing-page review, changes made or
why no page edit was needed, checks run and any unverified rendering. Do not claim a visual review
from a source-only check.

Follow [CLAUDE.md](../CLAUDE.md): one task branch and PR for shared work, one feature branch and PR per
paper. Update its reproduction record with changes to code, kernels or results. Use the repository's
integration branch if configured, and integrate into `main` only through a PR. Do not commit directly
to `main` or merge without the user's instruction.

# Research documentation alignment

## Implementation and purpose

This documentation job aligns the project around selecting, implementing and evaluating ABUS papers
on TDSC-ABUS 2023, followed by a Phase 3 study of Laplace–Beltrami representations. It adds a
[study index](../papers.md) and [research workflow](../research-workflow.md), aligns the README,
website metadata, Kaggle guide and contributor rules, and records the first paper's pending source.
Its notebook retains provisional findings and links to a detailed reproduction record.

## Requirements and verification

This job changes documentation and notebook Markdown only. It requires no data download or GPU run.
Validate local documentation links, notebook JSON/schema and preservation of code cells, outputs and
execution counts. Review the diff for unchanged application code, stored results, manifests and lockfiles.
No paper experiment is rerun or newly verified by this job.

## Decisions and status

- README comparisons for Barekatrezaei 2026 are deferred while finalized source, including CNN code,
  is awaited from the first author. Existing measurements remain in the notebook and reproduction
  record; this changes their presentation, not the stored experiment results.
- The index exposes missing per-paper guides instead of inventing provenance. Tan, Zhou and Wang
  guides remain work for their respective study PRs.
- Author verification is a separate status with a reviewable revision and stated scope. Outreach is
  planned; no email is sent by this job, and no author approval is claimed.
- Existing paths and notebook numbering are retained. Historical provenance is explicitly incomplete.
- Documentation changes use `docs/research-workflow`; integration into `main` requires a PR. No
  `develop` branch was present when this work started.

Validation completed for this change:

- Local Markdown links in the edited guides and notebook resolve.
- Notebook schema and website YAML parse successfully.
- All original notebook code cells, outputs and execution counts are unchanged.
- Every code cell in `2-barekatrezaei2026.ipynb` executed successfully against stored CSVs in the prepared
  CPU environment. The executed copy was kept outside the checkout; no training or cache build ran.
- `git diff --check` passed.

GPU reruns, full historical notebook execution and a website build are outside this documentation job.
New README documentation links target GitHub so they remain accessible from the project homepage,
whose Jekyll configuration excludes internal research guides.

## Landing-page follow-up

Added landing-page review to every PR's contributor and research workflow: align the public summary,
metadata, study status, reported results and relevant images with the supporting records. Source review
confirmed the existing README figures remain linked and have descriptive alt text. Corrected the
remaining README links to the excluded benchmark notebook, the Kaggle Markdown guide and the license
to target GitHub, so the shared README also works as landing-page content.

Static checks cover image integrity, local image/link targets and configured navigation anchors.
The current machine has no Ruby/Bundler installation, so a Jekyll build and rendered desktop/mobile
review were not performed. No layout/CSS changes or new scientific figures are introduced by this follow-up.

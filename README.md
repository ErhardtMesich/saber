# Saber

Code release for *A Closed-Loop Bio-Robotic Platform for Monitoring Liver Cancer Organoids
and Tumor-Microenvironment-Aware Drug Screening* (Cyborg and Bionic Systems).

## Overview

The platform treats continuous organoid monitoring as damage-budgeted information
acquisition. A dish of organoid cultures is observed by a robotic imaging and liquid-handling
head; every observation costs specimen integrity, and the question the release answers is
which observation to take next. The code implements the decision problem, not a product:

- the action set of Eq. (1) and the damage functional of Eq. (2)/(6), with the budget defined
  as the reference protocol's own damage over the same wall-clock window (Sec. 2.4); the
  equal-accuracy exposure-saving comparison is an extrapolation, not an iso-resource result;
- the receding-horizon planner of Eq. (7) and Algorithm 2, with the Lagrangian multiplier
  updated from the running damage and the action set pruned inside the control window;
- the two-statement theory layer: the interior-optimum result of Theorem 1, the constant-factor
  greedy guarantee of Proposition 1 and Corollary 1, and the parity result of Theorem 2, each
  recomputed from its own closed form rather than asserted;
- the perception front end of Sec. 2.2 (two adapted backbones, instance masks, trajectories,
  morphology embeddings), the four-axis variational estimator of Sec. 2.3 and Algorithm 1, the
  microenvironment-conditioned response head of Sec. 2.6, the coupled growth/oxygen twin of
  Sec. 2.7, and the off-policy estimator family of Sec. 3.7.

The original article's record-level cohort, trajectories and paired drug-response
outputs are unavailable in this release. The historical execution path is disabled.
Legacy monitor, estimate, transfer, twin, build_cohort and verification commands
require those missing records and therefore stop with an explicit error. They do
not reproduce the published results. The real-data experiment described below is
a separate feasibility test and cannot establish the paper's historical claims.

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # optional; the suite runs from src/ without it
```

```bash
conda env create -f environment.yml
conda activate saber
```

```bash
docker build -t saber .
docker run --rm saber python3 -m saber.cli.verify
```

## Real-data experiment

The article's Supplementary Table A1 lists the brain-organoid imaging dataset
[Schröter et al., Scientific Data (2024)](https://doi.org/10.1038/s41597-024-03330-z),
[Zenodo 10301912](https://doi.org/10.5281/zenodo.10301912), CC BY 4.0.
The original dataset comprises images and manually annotated segmentation masks.
It contains brain organoids, not liver cancer organoids, and no paired drug-response
or original study policy logs. Its relationship to the paper's A–E cohorts is
unknown. This experiment does not reproduce the paper's figures or metrics.

### Completed experiment: annotated area over time

The author's versioned `df_pred.csv` includes organoid identifiers, time points,
annotated areas and model-derived measurements. We downloaded the 1.4 MB table
at a fixed repository commit and verified its Git blob and SHA-256. It contains
1,407 rows; the experiment selects 36 measured records from four tracked brain
organoids, one per clone, imaged in Lab A. The 32 observations from days 2–25
(excluding the two day-12 embedding conditions) are used to fit one linear area
trend per organoid. Day 30 is held out and compared with both the fitted trend and
the last observed area. The unit is square micrometres; these are author-provided
annotation-derived areas, not images or newly measured clinical outcomes.

```bash
python3 scripts/download_brain_organoid_table.py  # needs gh and network access
python3 scripts/run_brain_organoid_growth.py
```

The downloaded table and provenance manifest are saved under
`data/public/brain_organoid_author_table/`. Selected records, individual forecasts
and summary metrics are saved under `results/real_brain_organoid_growth/`.
This small, fixed-subset feasibility test is not a liver cancer, drug response or
closed-loop validation and does not reproduce any manuscript outcome.

### Optional image-based experiment

To fetch six original Lab A images and their masks (two organoids, three time points
per organoid), plus the original dataset index, run:

```bash
python3 scripts/download_brain_organoid_subset.py
python3 scripts/run_brain_organoid_experiment.py
```

The downloader retrieves ZIP members using HTTP byte ranges, verifies their ZIP
CRC-32 and records SHA-256 hashes. It does not check the full archive MD5 because
it does not download the complete archive. The experiment checks those hashes and
evaluates a fixed Otsu segmentation baseline against author masks. This optional
experiment requires network access to Zenodo and has not yet run here because the
image download failed.

The prior `saber.cli` study commands are disabled because the required original
records and response labels are unavailable. Manuscript-number arithmetic is not
evidence from this real dataset.

## Legacy verification artifacts

The old verification artifacts have been removed. The historical cohort and
supporting records are unavailable, so the original reported results remain
unverified. The separate real-data experiment outputs are stored in
`results/real_brain_organoid_growth/`.

## Layout

```
configs/       panel, dish, perception, planner, belief, response, twin blocks and run files
scripts/       session, estimation, transfer, twin and verification launch scripts
src/saber/     spec, theory, twin, perception, belief, planner, response, evaluation, cohort,
               runtime and verification packages, plus the study orchestration module
tests/         per-module unit tests and the session smoke tests
```

## Licence

MIT; see `LICENSE`. Third-party packages and the resources the article's registry names are
listed with their licences and sources in `NOTICE`.

# CADET: Residualized Streaming Change Detection for Non-Stationary Reinforcement Learning with Probe-Access False-Alarm Guarantees

**Sooyoung Jang**<sup>1</sup>, **Siyeon Park**<sup>2</sup>,
**Sungpil Woo**<sup>1</sup>, and **Ahyun Lee**<sup>2</sup>

<sup>1</sup> Department of Computer Engineering, Hanbat National University,
Daejeon 34158, Republic of Korea<br>
<sup>2</sup> Department of Intelligent Media Engineering, Hanbat National University,
Daejeon 34158, Republic of Korea

Corresponding authors: Sungpil Woo (`woosungpil@hanbat.ac.kr`) and
Ahyun Lee (`ahyun@hanbat.ac.kr`).

> **Status:** Manuscript under review.

## Abstract

Online reinforcement learning changes its own monitoring signals, potentially
confusing policy updates with external shifts. CADET subtracts policy-update
feature differences evaluated on a frozen buffer and monitors the residual with
dispersion and sign-dynamics tests. The experiments use exact feature differences.
Finite-sample false-alarm guarantees require probe access, a common stationary
residual null, fixed or independently calibrated centers, and valid dependence
bounds. The delay guarantee additionally requires a retained clean reference.

This release contains **two confirmation experiments** using screened physics
residual readouts, empirically calibrated thresholds, and rolling references.
These calibrated results do not inherit the theoretical guarantees. A configuration
fixed before confirmation detects HalfCheetah gravity and Walker2d friction on fresh
seeds, but transfers to only one of four shift cases on previously unused
environments. These results characterize readouts and detectors within one MuJoCo
benchmark family, without establishing general superiority or deployment performance.

## Repository layout

```text
cadet/                         Detector, residualization, calibration, baselines
scripts/
  replay_round2_confirmations.py   Entry point for both saved confirmations
  exp007_channel_screened_cadet.py Channel screening and replay implementation
results/
  rev05_fresh_seed_window1530_confirmation/
  rev06_heldout_envs_window1530/
    screened_manifest.json     Settings and seeds (present in each run)
    screened_*.csv             Six reference tables per run
    traces/                    Arrays and trace_manifest.json per run
pyproject.toml                 Package and build metadata
requirements-lock.txt          Pinned replay and build dependencies
SHA256SUMS                     Checksums for distributed files
```

`cadet-access-r2-v1.0.0` supports only these two confirmations.
All replay data are included.

## Installation

Use **Python 3.12**, verified with **3.12.7**. In a Bash shell:

```bash
git clone https://github.com/pz1004/cadet-public.git
cd cadet-public
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
```

The [lock file](requirements-lock.txt) pins NumPy 1.26.4, pandas 2.2.2,
SciPy 1.13.1, and their replay/build dependencies. For an extracted release, start
at environment creation from its package root.

## Included data

| Confirmation | Environments | Saved arrays |
|---|---|---:|
| [Fresh seeds](results/rev05_fresh_seed_window1530_confirmation/) | HalfCheetah-v5, Walker2d-v5 | 200 |
| [Held-out environments](results/rev06_heldout_envs_window1530/) | Hopper-v5, Ant-v5 | 200 |

Both use horizon **600**, change time **300**, scale/dynamics windows **15/30**,
and warm-up **80**. Gravity is multiplied by **1.45** and friction by **1.8** in
their respective shift scenarios. The empirical calibration target is a **0.05**
run-level false-alarm rate; it is not a guaranteed population rate.

Each environment contributes 20 no-shift calibration traces, 20 separate no-shift
held-out traces, and 20 evaluation traces for each of no shift, gravity, and
friction. Together: **400 arrays**, **four manifests**, and **12 reference CSVs**.
Manifests retain settings, split assignments, and seeds.
Fresh-seed confirmation uses seeds disjoint from configuration selection;
Hopper and Ant were absent from readout selection.

## Reproducing the results

From the installed package root, verify the distribution and run both confirmations:

```bash
sha256sum -c SHA256SUMS
python scripts/replay_round2_confirmations.py
```

Replay reads settings from the saved manifests and requires **no MuJoCo
installation**. It compares **all 12 reference CSVs exactly**: selected channels,
run maxima, empirical thresholds, held-out false-alarm rates, per-run results,
and summaries for each confirmation. A missing required array causes an explicit
failure without regeneration.

Generated tables are written beneath `build/round2/replay/`, one subdirectory per
confirmation. `build/round2/replay/verification.json` records settings, trace counts,
and exact table comparisons after both confirmations pass. Reference results are
preserved. Downloaded archives also have a separate `.tar.gz.sha256` checksum.

## Results

Counts are detections out of **20 shifted runs**. Delays are mean steps after the
change **among detected runs**, rounded to two decimals; “—” means no detections.
MMD uses the same screened channels. Source summaries:
[fresh seeds](results/rev05_fresh_seed_window1530_confirmation/screened_summary.csv)
and [held-out environments](results/rev06_heldout_envs_window1530/screened_summary.csv).

| Environment | Shift | CADET detections | CADET delay | MMD detections | MMD delay |
|---|---|---:|---:|---:|---:|
| HalfCheetah-v5 | Gravity | 20/20 | 15.00 | 20/20 | 16.20 |
| HalfCheetah-v5 | Friction | 0/20 | — | 1/20 | 31.00 |
| Walker2d-v5 | Gravity | 0/20 | — | 0/20 | — |
| Walker2d-v5 | Friction | 19/20 | 34.11 | 19/20 | 35.68 |
| Hopper-v5 | Gravity | 0/20 | — | 0/20 | — |
| Hopper-v5 | Friction | 18/20 | 54.17 | 20/20 | 13.95 |
| Ant-v5 | Gravity | 0/20 | — | 0/20 | — |
| Ant-v5 | Friction | 0/20 | — | 0/20 | — |

Both detectors have **0/20 alarms in each environment's no-shift evaluation**.
The separate held-out no-shift splits have **1/20 for CADET on HalfCheetah** and
zero elsewhere; all pass the empirical admissibility gate. On shifted runs,
CADET has **one pre-change alarm** for HalfCheetah friction; all other pre-change
counts are zero. That alarm is not a detection. See the
[fresh-seed](results/rev05_fresh_seed_window1530_confirmation/screened_heldout_far.csv)
and [held-out-environment](results/rev06_heldout_envs_window1530/screened_heldout_far.csv)
false-alarm tables alongside the summaries.

CADET detects none of five tested environment/shift combinations. MMD detects more
Hopper friction shifts with a shorter mean detected delay.

## Citation

```bibtex
@unpublished{jang2026cadet,
  title  = {{CADET}: Residualized Streaming Change Detection for Non-Stationary
            Reinforcement Learning with Probe-Access False-Alarm Guarantees},
  author = {Jang, Sooyoung and Park, Siyeon and Woo, Sungpil and Lee, Ahyun},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgement

This research was supported by the Regional Innovation System & Education (RISE)
program through the Daejeon RISE Center, funded by the Ministry of Education (MOE)
and Daejeon Metropolitan City, Republic of Korea (2026-RISE-06-002).

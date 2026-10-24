# CADET

CADET provides streaming change detection for non-stationary reinforcement
learning signals. It combines policy-update residualization (PUR) with scale
and sign-dynamics monitors. The package includes baseline detectors, synthetic
streams, empirical calibration tools, and optional MuJoCo trace generation.

## Installation

Requires **Python 3.10 or newer**. From a terminal:

```bash
git clone https://github.com/pz1004/cadet-public.git
cd cadet-public
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
```

The editable install makes local package changes available immediately.
Run the commands below from the repository root with the environment active.

## Usage

### Synthetic quick start

Run a small synthetic experiment:

```bash
python scripts/run_experiments.py --quick
```

This generates streams with no change, scale changes, dynamics changes, and
mixed changes. Outputs are written to `results/quick/`, including:

- `raw_results.csv`: results for each detector and run.
- `summary.csv` and `summary.md`: aggregate metrics.
- `ablation_summary.csv`: detector ablation summaries.
- `detectability_audit.csv`: diagnostic signal separation scores.
- `trace_manifest.json`: stream and experiment configuration.

The CLI uses nominal alpha-spending thresholds by default. These do not
establish an empirical false-alarm rate. Add `--match-far` to tune detector
alphas on no-change calibration runs; assess false alarms on held-out runs.
The detectability audit is diagnostic and is not a calibrated null test.

For run counts, horizons, calibration settings, and output paths, see:

```bash
python scripts/run_experiments.py --help
```

### Optional MuJoCo example

Install the Gymnasium MuJoCo dependencies in the same environment:

```bash
python -m pip install -e '.[mujoco]'
```

Generate a small HalfCheetah trace suite with no change and a gravity change:

```bash
python scripts/run_experiments.py --source mujoco --quick \
  --mujoco-envs HalfCheetah-v5 --outdir results/mujoco_quick
```

Results go to `results/mujoco_quick/`, with generated traces in its `traces/`
subdirectory. The default readout uses a lightweight online actor-value probe.
For empirical statistic thresholds and channel screening, inspect the dedicated
runners' options:

```bash
python scripts/exp006_mujoco_empirical_calibrated.py --help
python scripts/exp007_channel_screened_cadet.py --help
```

## Testing

```bash
python -m pytest -q
```

## Directory layout

```text
cadet/           Core detectors, residualization, calibration, and stream tools
scripts/         General runner and numbered runners 001–004, 006, and 007
tests/           Self-contained package and runner tests
pyproject.toml   Package metadata and dependencies
results/         Generated experiment output (created on demand; ignored by Git)
```

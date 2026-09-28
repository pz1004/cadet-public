#!/usr/bin/env python
"""Replay both confirmations with every setting read from the saved manifests.

No MuJoCo regeneration occurs: missing arrays are errors. All six CSV outputs
(channels, maxima, thresholds, held-out FAR, per-run results, summary) must agree.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.exp007_channel_screened_cadet import ScreenedReplaySettings, run

RUNS = ('rev05_fresh_seed_window1530_confirmation', 'rev06_heldout_envs_window1530')
TABLES = ('screened_channels', 'screened_run_max_statistics',
          'screened_empirical_thresholds', 'screened_heldout_far',
          'screened_raw_results', 'screened_summary')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/round2/replay')
    args = parser.parse_args()
    # Check both confirmations before replay can reach the generic trace generator.
    for name in RUNS:
        trace_dir = args.results_root / name / 'traces'
        traces = json.loads((trace_dir / 'trace_manifest.json').read_text())['traces']
        missing = [trace_dir / row['filename'] for row in traces
                   if not (trace_dir / row['filename']).is_file()]
        if missing:
            raise FileNotFoundError(
                'Missing required confirmation arrays; regeneration is disabled: '
                + ', '.join(str(path) for path in missing))
    records = []
    for name in RUNS:
        source = args.results_root / name
        manifest = json.loads((source / 'screened_manifest.json').read_text())
        config = {f.name: manifest[f.name] for f in fields(ScreenedReplaySettings)
                  if f.name in manifest}
        config.update(outdir=args.output / name, load_traces=source / 'traces', save_traces=None)
        for key in ('mujoco_envs', 'scenarios', 'arms', 'screened_baselines',
                    'screen_include_regex', 'screen_exclude_regex'):
            config[key] = tuple(config[key])
        print(f'Replaying {name}', flush=True)
        run(ScreenedReplaySettings(**config))
        comparisons = []
        for table in TABLES:
            expected = pd.read_csv(source / f'{table}.csv')
            actual = pd.read_csv(config['outdir'] / f'{table}.csv')
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
            comparisons.append({'table': table, 'rows': len(actual), 'exact': True})
        records.append({'run': name, 'settings': manifest, 'comparisons': comparisons,
                        'trace_arrays': len(list((source / 'traces').glob('*.npz')))})
        print(f'PASS: all {len(TABLES)} tables agree exactly', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__':
    main()

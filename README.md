# CADET confirmation replay package

`cadet-access-r2-v1.0.0` supports exact replay of two saved CADET confirmation
experiments. It is staged for author review; public publication is pending.
The repository is https://github.com/pz1004/cadet-public. Package files have not
yet been pushed and will be made available there before submission.

Included confirmations:

- `rev05_fresh_seed_window1530_confirmation`: HalfCheetah-v5 and Walker2d-v5.
- `rev06_heldout_envs_window1530`: Hopper-v5 and Ant-v5.

Together they contain 400 saved trace arrays, original experiment and trace
manifests, and 12 reference CSVs. The manifests preserve the calibration,
held-out, and evaluation settings and seeds. Both experiments use horizon 600,
change time 300, scale/dynamics windows 15/30, and warm-up 80. Detector code,
arrays, manifests, and reference results are unchanged from the development
repository. This release supports only these two confirmations; the full search
record and manuscript sources remain in the development repository.

Use Python 3.12 (verified with 3.12.7). From the extracted package root, verify
checksums, create an isolated environment, install the pinned dependencies and
package, then run both confirmations:

```bash
sha256sum -c SHA256SUMS
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python scripts/replay_round2_confirmations.py
```

Replay requires no MuJoCo installation. It reads the saved arrays and checks all
12 saved CSVs exactly: selected channels, run maxima, empirical thresholds,
held-out false-alarm rates, per-run results, and summaries for each confirmation.
A missing required array causes an explicit failure rather than regeneration.
Outputs and `verification.json` are written to `build/round2/replay/`; reference
results remain unchanged. The archive has a separate `.tar.gz.sha256` checksum.

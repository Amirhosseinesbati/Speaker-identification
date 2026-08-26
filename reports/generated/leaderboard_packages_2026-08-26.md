# Leaderboard packages — 2026-08-26

## Recommended submission order

1. `submission_campp_historical_control.zip`
2. `submission_campp_ecapa_80-20_centroid.zip`

The first package is the calibration/control anchor. Submit it first so its
score can confirm that the historical CAM++ recipe has been reproduced before
interpreting the CAM++/ECAPA experiment.

## Historical CAM++ control

- ZIP: `submission_campp_historical_control.zip`
- Size: 86.907 MiB
- SHA-256: `b6781261e3252250f00a640f6a46e5f06609f4ac1640efdb1b05c29d8aef09e4`
- Checkpoint source: `checkpoints/modelrigestry/campp_best (5).pt`
- Checkpoint SHA-256 prefix: `ff5108b0e037130d`
- Known-centroid SHA-256 prefix: `59a0339fcd040dcd`
- Unknown-cluster centroids: disabled
- Decision: alpha=0.3, kappa=16, tau=0, lambda_unknown=0.7, T=1
- Reference validation Macro-F1: 0.9622757380
- TitaNet: absent

## CAM++ / ECAPA centroid ensemble

- ZIP: `submission_campp_ecapa_80-20_centroid.zip`
- Size: 287.132 MiB
- SHA-256: `2f5a168d6a371555b9e20e71e801fd7b1b9069e3cce045fb148297163786c502`
- CAM++ source: `checkpoints/modelrigestry/campp_best (6).pt`
- CAM++ SHA-256 prefix: `e461cf867540c872`
- ECAPA source: `checkpoints/modelrigestry/ecapa_best.pt`
- ECAPA SHA-256 prefix: `5caf3ec536d32b27`
- Probability/centroid weights: CAM++=0.8, ECAPA=0.2
- CAM++ unknown-cluster centroids: enabled (554)
- Decision: alpha=0.3, kappa=16, tau=0, lambda_unknown=0.65, T=1
- Reference validation Macro-F1: 0.9649041423
- TitaNet: absent

## Verification

- Both ZIPs passed `scripts/verify_submission.py`, including extraction from a
  foreign working directory, eight-file inference smoke test, stdout policy,
  and output CSV validation.
- Full project test suite: 250 passed, 41 skipped, 2 pre-existing scheduler
  warnings.
- Both packages are below the 1 GiB leaderboard limit.

## Rebuild commands

```powershell
$env:UV_CACHE_DIR='.uv-cache-codex'
$env:PYTHONIOENCODING='utf-8'
uv run --no-sync python scripts/build_submission.py --fusion-config configs/submissions/campp-historical-control.json --zip-output submission_campp_historical_control.zip
uv run --no-sync python scripts/build_submission.py --fusion-config configs/submissions/campp-ecapa-80-20-centroid.json --zip-output submission_campp_ecapa_80-20_centroid.zip
```

The builder now supports package-specific known centroid sources, explicit
unknown-centroid inclusion, and package-specific decision configs. This avoids
silently pairing a historical checkpoint with whichever global calibration
files happen to be active in `data/processed`.

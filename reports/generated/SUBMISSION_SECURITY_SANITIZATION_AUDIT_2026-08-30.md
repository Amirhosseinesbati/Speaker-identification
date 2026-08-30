# Submission Security Sanitization Audit — 2026-08-30

## Scope

The current record-setting leaderboard archive was re-audited locally without
modifying or deleting it:

- path: `data/artifacts/iaaa_campp_lme20_pcm_recovery_20260829.zip`
- leaderboard Macro-F1 reported by the user: `0.9667174284505605`
- size: `94,231,494` bytes (`89.866 MiB`)
- SHA256: `3653d0f4e54433f4096a521d814d5e606c5b9314fefa986b315b7623143a7494`
- archive entries: `132`
- full entry stream/CRC read: passed
- one-gigabyte competition limit: passed

## Finding

The archive unintentionally contained ModelScope cache authentication state at
`weights/campp/credentials/session`.  Its contents were neither read nor
printed.  No packaged Python, JSON or Markdown file references this path, and
the offline CAM++ snapshot and trained checkpoint are both already present.

Because the original archive has already been uploaded to an external
leaderboard, the corresponding ModelScope/cache session should be considered
potentially exposed and rotated or revoked by the account owner even though it
may be expired or non-privileged.  No credential rotation was attempted by the
campaign tooling.

## Sanitized operational copy

A new local archive was produced by removing only the single cache-session
entry; the record-setting original was preserved byte-for-byte:

- path:
  `data/artifacts/iaaa_campp_lme20_pcm_recovery_sanitized_20260830.zip`
- size: `94,155,173` bytes (`89.793 MiB`)
- SHA256: `1b0e715bc215a01e34b6ae4f4eae424c0a9685f7e7de311b8c1d445c1ac96692`
- archive entries: `131`
- credential-like entries: `0`
- full entry stream/CRC read: passed
- one-gigabyte competition limit: passed

Every non-session entry was compared by archive path, uncompressed length and
SHA256.  All `131/131` non-sensitive payloads are identical and no expected
entry is missing.

## Execution smoke test

The sanitized archive was extracted locally and its official leaderboard
entry point was run, offline, on the three bundled CAM++ example WAV files:

`python submission.py --data-dir <bundled-examples> --predictions-file-path <csv>`

The command exited `0`; the CSV had exactly the required
`audio_file,speaker_id` columns, three nonblank rows, and no network or
credential dependency.  The examples are out-of-competition VoxCeleb audio,
so all three being predicted as `unknown` is a structurally valid smoke result,
not a leaderboard evaluation.

This sanitized archive is packaging-equivalent to the scored candidate and is
the only copy that should be reused.  It is not a new scientific candidate and
does not justify another leaderboard submission by itself.

## Preventive code change

`scripts/build_submission.py` now:

1. removes credential, secret, hub-lock and session/token cache paths from the
   assembled submission tree;
2. fails closed before ZIP creation if any such path remains; and
3. retains the existing one-gigabyte size guard.

Regression tests cover detection, cleanup, preservation of a benign
`src/session.py`, and the existing size boundary.

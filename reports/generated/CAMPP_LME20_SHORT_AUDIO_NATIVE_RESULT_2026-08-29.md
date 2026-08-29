# CAM++ LME20 native-length short-audio result — 2026-08-29

## Decision

**Rejected.**  Native-duration inference for files shorter than eight seconds
does not generalise across the three fixed OOF folds.  The locked pad-mode
LME20 package remains unchanged.

## Metrics

| Scope | Macro-F1 delta | Known Accuracy delta | OOD-F1 delta |
|---|---:|---:|---:|
| Fold 0 | -0.0015236 | -0.0011211 | -0.0032595 |
| Fold 1 | +0.0006862 | +0.0022548 | +0.0019722 |
| Fold 2 | -0.0039820 | 0.0000000 | -0.0021849 |
| Aggregate | -0.0024755 | +0.0004488 | -0.0011632 |

The locked pad baseline reproduced exactly at Macro-F1
`0.9633564052154656`; native length scored `0.9608809278201754`.  Across all
4447 OOF files it changed 45 predictions, rescued 8 pad errors and introduced
14 errors.  Among the 182 short validation files, pad classified 112 correctly
and native length classified 111 correctly (4 short-file rescues, 5 short-file
regressions).

## Provenance

- Enrollment and validation short rows used the same native-duration rule.
- All long rows were copied from pad artifacts and remained bitwise identical.
- Native forward lengths ranged from 16,384 samples (1.024 seconds) to 126,976
  samples (7.936 seconds).
- There was no threshold, duration cut-off, blend or leaderboard selection.
- Cache SHA256 values:
  - Fold 0: `1a0d65a9f796c334ec5700b41b1c636b35db2e096486588a75cc8a653887c306`
  - Fold 1: `14e1aacf6d518b116395c397e4bdf327e4af0c9a29ef462be97734c9c0abdfc8`
  - Fold 2: `17015291fe60074ab7a9c7d3b6fc9807ffbc4db809750c60043110cc28f9c441`
- Result JSON SHA256:
  `6c4a13fadc2606bffd6f79f2f3100d4393240d8923ca922522e3bfd439e49350`.

## Interpretation

Both naive alternatives to zero padding now fail: periodic repetition and
native-duration inference.  Fold 1 again benefits, but Folds 0 and 2 regress,
especially in OOD-F1.  This consistent topology says the Fold-1 domain slice
has a real duration mismatch, while a global short-audio transform is not
portable.  Selecting a Fold-1-only rule or learning a duration threshold from
these OOF outcomes would be post-hoc and is rejected.

The result shifts the next hypothesis away from waveform-length replacement.
The scientifically cleaner direction is to keep the validated eight-second
forward path and test robust aggregation of the *existing* multiple temporal
views or training-side prototype evidence, with every rule fixed before its
held-out evaluation.

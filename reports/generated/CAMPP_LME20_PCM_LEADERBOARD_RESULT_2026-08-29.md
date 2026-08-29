# CAM++ LME20 + PCM recovery leaderboard result — 2026-08-29

## Official result reported by the competition panel

- Task: `scoring`
- Panel timestamp as displayed: `08/29/2026, 18:13:05`
- Accuracy: `0.9667036625971143`
- Macro-F1: `0.9667174284505605`
- Rank in the user-provided snapshot: `7`
- Package:
  `data/artifacts/iaaa_campp_lme20_pcm_recovery_20260829.zip`
- Package size: `94231494` bytes (`89.9 MiB`)
- Package SHA256:
  `3653d0f4e54433f4096a521d814d5e606c5b9314fefa986b315b7623143a7494`

The locked three-Fold OOF Macro-F1 is `0.9633564052154656`; the observed
leaderboard score is higher by `0.0033610232350949`.  This is strong external
evidence for the complete packaged stack, but it is not a controlled
single-variable comparison that can allocate the gain specifically to PCM
recovery, LME20, or distribution differences.  The leaderboard result is an
external anchor, not a source for tuning thresholds, blends or epochs.

The gap to the user-provided first-place score `0.972643` is
`0.0059255715494395`.  The next internal candidate must still pass predefined
three-Fold OOF direction and Known/OOD guardrails before any new package is
built or submitted.

## User-provided top-10 snapshot

| Rank | Team | Macro-F1 | Count shown | Time shown |
|---:|---|---:|---:|---|
| 1 | Trench Battle Hero | 0.972643 | 43 | 8/29/2026, 12:30:12 AM |
| 2 | آنید | 0.970224 | 37 | 8/29/2026, 10:29:16 AM |
| 3 | lian | 0.969900 | 35 | 8/29/2026, 5:28:10 PM |
| 4 | دیپ | 0.969703 | 39 | 8/29/2026, 3:19:14 PM |
| 5 | IRAScience | 0.968086 | 61 | 8/28/2026, 1:24:48 PM |
| 6 | اکام | 0.966951 | 10 | 8/29/2026, 9:38:56 AM |
| 7 | Madia | 0.966717 | 16 | 8/29/2026, 5:25:38 PM |
| 8 | Kavian | 0.965817 | 13 | 8/18/2026, 8:31:02 PM |
| 9 | TIKCUS | 0.963930 | 51 | 8/26/2026, 9:37:38 AM |
| 10 | Rofagha AI | 0.961721 | 46 | 8/28/2026, 10:59:40 PM |

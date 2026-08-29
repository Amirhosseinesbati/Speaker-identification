# CAM++ Raw/EMA LME20 snapshot ensemble result — 2026-08-29

## Decision

Rejected.  Fixed `50/50` Raw/EMA evidence averaging does not improve the
externally validated Raw LME20 policy and violates both aggregate guardrails.
No submission ZIP is authorised.

## Locked three-fold result

| Fold | Raw Macro-F1 | Raw/EMA Macro-F1 | Delta | Known delta | OOD-F1 delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9611456663 | 0.9600406509 | -0.0011050153 | -0.0033632287 | -0.0050529598 |
| 1 | 0.9589145283 | 0.9593650172 | +0.0004504889 | 0.0000000000 | +0.0013685312 |
| 2 | 0.9406726612 | 0.9410455173 | +0.0003728561 | 0.0000000000 | 0.0000000000 |
| **OOF** | **0.9633564052** | **0.9630353626** | **-0.0003210426** | **-0.0013464991** | **-0.0012288623** |

The ensemble rescued seven Raw errors and introduced four errors across 20
changed predictions, so its ordinary accuracy increased by `+0.0006746121`.
Macro-F1 nevertheless decreased because the changes were not class-balanced.
The dominant failure is Fold 0.

## Error-topology interpretation

Raw aggregate errors contain 87 `known->unknown`, 19
`known->wrong-known`, and 25 `unknown->known` cases.  The equal snapshot
ensemble changes those counts to 99, 10, and 19.  EMA therefore contains real
identity and OOD complementarity, but the full evidence average buys fewer
wrong-known and unknown false accepts by rejecting too many known speakers.

This result rejects full Raw/EMA evidence fusion.  It motivates one narrower,
fully pre-registered follow-up that has not yet been scored: preserve every Raw
known/unknown decision and use the fixed Raw/EMA evidence only to rerank the
known identity.  Such a candidate cannot trade known identity errors for OOD
rejections and will be evaluated without parameter selection.

## Provenance

- implementation/preregistration commit: `a1c7a87632ad`
- result JSON: `reports/generated/campp_raw_ema_lme20.json`
- result JSON SHA256:
  `caba17da1fc02cd47c186a0838bf94c48c153ccb9380b15b60cc201cceae88d5`
- 4,447 unique, non-overlapping OOF files
- Raw self-ensemble reproduced the locked probabilities/decisions to `1e-8`
- worker limited tests: 5 passed
- local full suite: 159 passed, 8 skipped
- leaderboard used for parameter selection: no

All three EMA cache payloads and metadata files were SHA256-verified on the
worker.  Their hashes are embedded in the result JSON provenance.  GPU
extraction completed without OOM, NaN, split drift, or file-order mismatch.

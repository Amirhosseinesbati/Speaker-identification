# Speaker channel-mismatch literature ledger — 2026-08-29

## Purpose

This ledger records which literature findings change the IAAA campaign decision
tree.  Papers motivate hypotheses; locked OOF evidence and Known/OOD guardrails
remain authoritative.  Leaderboard results are external anchors and are never
used to choose thresholds, fusion weights, epochs, or augmentation strength.

## Evidence mapped to local decisions

| Evidence | Relevant finding | Local evidence | Campaign decision |
|---|---|---|---|
| VoxWatch open-set benchmark ([DOI](https://doi.org/10.48550/arXiv.2307.00169)) | Adaptive score normalization is not guaranteed to help open-set speaker identification, while calibration and genuinely complementary score fusion can help. | AS-Norm was rejected by locked OOF; fixed LME20/PCM transferred from `0.9633564052` OOF to `0.9667174285` leaderboard Macro-F1. | Keep LME20/PCM as the operational baseline.  Do not revive score normalization without a new, leakage-free hypothesis. |
| Comprehensive speaker augmentation study ([DOI](https://doi.org/10.21437/Interspeech.2024-2478)) | Speed and vocal-tract perturbations can create speaker variation rather than merely preserve channel variation. | The residual error topology is dominated by Known/OOD boundary errors, so identity-corrupting augmentation can increase known-speaker rejection. | In the active treatment, pitch shift is disabled and time stretch is rare and restricted to `0.95..1.05`. |
| Noise/reverberation robustness via training ([DOI](https://doi.org/10.1109/EISIC.2015.20)) and multi-channel training ([DOI](https://doi.org/10.21437/INTERSPEECH.2019-1437)) | Training on additive noise and RIR/channel variability can reduce mismatch degradation. | MUSAN and RIR assets are already local to the worker; no data download or provenance expansion is required. | Test one single-variable CAM++ Fold-0 treatment with stronger speaker-preserving MUSAN/RIR exposure. |
| Domain-weighted low-resource transfer ([DOI](https://doi.org/10.1186/s13636-024-00385-z)) | Uncontrolled fine-tuning under domain mismatch can degrade performance; domain-weighted adaptation can be safer. | The fixed-30 CAM++ LMFT experiment did not beat its warm-start checkpoint. | Do not repeat generic LMFT.  A future adaptation experiment must define a domain proxy, weighting rule, held-out gate, and stop rule before training. |
| Curriculum learning for speaker verification ([arXiv](https://arxiv.org/abs/2203.14525)) and the NIST SRE21 overview ([DOI](https://doi.org/10.21437/Odyssey.2022-45)) | Gradually increasing augmentation difficulty can improve speaker representations, and long-duration fine-tuning was part of strong mismatched-domain systems. | The active channel-robust treatment is deliberately harder than Control; after unfreezing, its EMA and Raw curves are still rising rather than plateauing. | Do not stop the active run merely because early post-unfreeze epochs lag.  Let its preregistered patience/timeout act, then use the terminal slope and saved optimiser/scheduler state to decide whether a state-preserving continuation is scientifically warranted. |
| Adversarial data augmentation ([DOI](https://doi.org/10.1145/3638884.3638917)) and ASVspoof5 augmentation ablations ([paper](https://www.isca-archive.org/asvspoof_2024/xie24_asvspoof.pdf)) | Vanilla augmentation can leave augmentation-specific residuals; overly broad or overly frequent transforms do not improve every task, while moderate frequency masking can outperform stronger combinations. | The current treatment uses several simultaneous channel transforms and has not yet beaten Control terminal performance. | Extra epochs are evidence collection, not a presumption that more is always better.  If terminal performance fails, prefer a preregistered curriculum or single-factor ablation over increasing all augmentation probabilities again. |
| Robust Channel Speaker Learning ([arXiv](https://arxiv.org/abs/2406.10956)) | A radio-domain system explicitly models bandwidth variation and unknown noise rather than treating every mismatch as generic additive noise. | The active treatment covers codec, RIR and waveform noise but has no explicit telephony/radio bandwidth transform or manifold-noise objective. | If the terminal error topology remains channel-sensitive, test controlled bandwidth restriction as one isolated training factor before adding another multi-component framework. |
| Noise-adaptive warm-up for speaker verification ([DOI](https://doi.org/10.21437/Interspeech.2024-1630)) | Teacher-student consistency between clean and noisy views is paired with an angular/prototypical objective to reduce speaker-information distortion from robust training. | At epochs 42--47, the harder treatment sometimes beats matched Control in Raw aggregation while EMA remains slower and Known/OOD balance still oscillates. | If stronger augmentation improves Raw representation but violates terminal guardrails, clean/augmented embedding consistency is a better-defined next hypothesis than simply increasing augmentation probability or training time. |
| Asymmetric clean-segment-guided learning ([arXiv](https://arxiv.org/abs/2309.04265)) | An explicit clean view can mitigate speaker-information loss from noisy/RIR views; its ablation improved the reported speaker-verification objective even without the paper's hard-negative weighting. | The active supervised run has labels and already uses strong augmentation, but it has no paired clean-view constraint. Raw and EMA improve at different rates, consistent with useful adaptation plus possible augmentation residual. | If the active terminal candidate fails, the cheapest representation follow-up is one paired clean/aug embedding-consistency term while keeping the supervised ArcFace/OOD objectives fixed. Do not import the paper's full self-supervised or hard-negative system as a bundled change. |
| Augmentation-adversarial speaker learning ([arXiv](https://arxiv.org/abs/2007.12085)) | Contrasting segments from the same utterance can retain shared channel cues; explicitly suppressing augmentation information encourages speaker-discriminative, channel-invariant embeddings and improved VoxCeleb/VOiCES robustness. | The active dataset produces eight independently cropped and augmented windows, so ordinary same-label supervision does not require the embedding to agree with a clean view of the identical crop. | This independently supports a paired clean/aug consistency ablation if the current treatment fails.  Use the supervised ArcFace/OOD model and one predefined cosine-consistency coefficient; do not add an adversarial domain classifier and a consistency term in the same Run. |
| Gradient regularization for noise robustness ([DOI](https://doi.org/10.21437/Interspeech.2021-1216)) | Aligning gradients from clean utterances and their noisy counterparts reduced speaker-irrelevant noise directions and improved both seen and unseen noisy conditions. | The active strong-augmentation treatment is still learning but retains a small validation-loss penalty relative to matched Control, which is compatible with useful robustness plus augmentation-specific gradients. | Treat clean/noisy gradient alignment as mechanistic corroboration, not the first implementation: its sequential inner optimization and gradient-level objective are costlier and less isolated than a paired embedding-consistency term. |
| Label-efficient speaker VICReg ([DOI](https://doi.org/10.21437/Interspeech.2022-802)) | Two independently augmented views are pulled together, but variance and covariance terms are needed to prevent collapse and redundant dimensions; the reported VICReg speaker system outperformed its InfoNCE-only comparison. | A plain cosine-consistency loss would preserve same-crop invariance but could over-compress the already discriminative 446-way embedding space, especially with batch 48 and an open-set OOD head. | If paired consistency is tested, retain ArcFace and OOD supervision and preregister an embedding-variance/covariance diagnostic guardrail.  Do not transplant the full self-supervised projector or jointly tune several VICReg coefficients. |
| Multi-Head Multi-Mode distillation ([DOI](https://doi.org/10.21437/Interspeech.2024-360)) | The authors found embedding-level contrastive learning useful only in early training and dynamically stopped that auxiliary term while continuing knowledge distillation. | The active treatment will add a fixed clean/aug cosine term to an already discriminative supervised model.  A longer horizon may be needed for the harder objective, but the same auxiliary pressure may also become unhelpful late in training. | Preserve the locked fixed-weight 120-epoch A/B test, but do not equate extra epochs with benefit.  The preregistered epochs 101--120 tail slope, best-epoch location, Known/OOD balance and spread guardrail decide whether a separately matched extension is warranted.  A future scheduled-off consistency ablation would require a new preregistration, never a post-hoc edit to this Run. |
| Self-Supervised Positive Sampling ([DOI](https://doi.org/10.21437/Interspeech.2025-183)) | Same-utterance positive pairs can retain recording-channel information; selecting same-speaker positives from different recording conditions reduced intra-speaker variance and improved both SimCLR and DINO speaker verification. | The current paired treatment compares clean and augmented views of the identical crop.  Worker EDA confirms all 446 known speakers have at least five distinct files: 439 have 5, five have 6, one has 9 and one has 20; the single collapsed unknown label has 2,275 files. Direct session/channel metadata is absent, but distinct known files are an available recording proxy. | If same-crop consistency fails or only marginally helps, the next isolated representation hypothesis is a known-only speaker-balanced sampler with cross-file positive pairs.  Use different files as a declared recording proxy, leave the OOD objective unchanged, and test against a matched sampler control; do not add SSPS clustering, a memory queue and a new loss in one bundled run. |
| Codec/domain robustness evaluation ([DOI](https://doi.org/10.21437/Interspeech.2025-2167)) | ECAPA-family and other deep speaker embeddings all degrade under sampling-rate and low-bitrate codec mismatch, indicating dependence on high-frequency information rather than architecture-specific immunity. | The current treatment includes MP3 but no isolated resampling/bandwidth factor; a previous complementary encoder did not clear its standalone gate. | Do not switch encoder merely because it is newer. If terminal error analysis remains codec-sensitive, test one predefined bandwidth/codec factor against the same CAM++ baseline before considering another backbone. |
| NEC-TT SRE18 system ([DOI](https://doi.org/10.21437/Interspeech.2019-1517)) | Diverse augmentation plus mixed-bandwidth training strengthened the embedding extractor, while CORAL/CORAL+ adapted the PLDA backend under mismatch. | The competition has no trusted target-domain labels or PLDA backend; the record package already uses a validated open-set prototype decision layer. | Treat mixed-bandwidth training as corroboration for one isolated bandwidth ablation.  Do not transplant CORAL/PLDA into the current open-set pipeline without a separate leakage-free contract. |
| DB-PMAE local/prototypical matching ([DOI](https://doi.org/10.21437/Interspeech.2024-897)) | Local patch correspondence and learned prototypes can improve domain robustness, but require a dual-branch masked-autoencoder pretraining stage plus SID/SV multi-task fine-tuning. | Our residual ceiling calls for representation change, but the remaining campaign budget and existing CAM++ evidence favour a controlled ablation over a new pretraining stack. | Keep local correspondence as a later architectural research direction; it is not a justified immediate Run while paired consistency and isolated bandwidth remain untested. |
| Extended variability modelling ([DOI](https://doi.org/10.21437/INTERSPEECH.2017-1586)) and open-set mismatch work ([DOI](https://doi.org/10.21437/Interspeech.2009-395)) | Duration/session/channel-aware backends and condition-adjusted normalization can help under mismatch. | Direct session/channel metadata is absent; quality-aware and normalization candidates already violated local guardrails. | Treat condition-aware prototype banks as a backup research direction, not an authorised run.  Any parameters must be selected leave-one-fold-out from training folds only. |
| Contrastive adversarial domain adaptation ([DOI](https://doi.org/10.1109/TNNLS.2020.3044215)) | Separating speaker-discriminative and domain-invariant objectives can improve mismatched speaker recognition. | No reliable channel labels exist and this is a multi-component architectural change. | Keep as a later high-cost hypothesis only after cheaper channel augmentation and prototype conditioning are resolved. |
| DINO speaker-verification curriculum ([paper](https://www.isca-archive.org/interspeech_2023/heo23b_interspeech.pdf)) | The authors make training progressively harder by increasing the fraction of augmented utterances; augmentation curriculum alone improved the reported baseline from `6.70%` EER to `6.35%` in its better course, but its interaction with the data curriculum was not uniformly positive. | Our fixed-hard treatment produced a late best at epoch 81, so slow adaptation is real, but curriculum is not guaranteed to beat full augmentation. | If the current treatment and its locked continuation fail, test one fixed augmentation-frequency ramp against an otherwise matched fixed-hard control; do not tune ramp breakpoints on Fold 0. |
| TalTech SdSV AugMix ablation ([paper](https://www.isca-archive.org/interspeech_2020/alumae20_interspeech.pdf)) | Stochastic clean/augmented mixing was retained, while JSD prediction consistency was dropped because it gave no material benefit and made training about `2.5x` slower. | Paired consistency is plausible but not a free win, especially with eight windows and a fixed campaign budget. | Keep the already preregistered cosine-consistency candidate as a matched A/B ablation with throughput and representation-variance diagnostics; reject it if the auxiliary cost or guardrails fail. |

## Active experiment and interpretation boundary

The active decision experiment is now the matched long-horizon pair declared in
`CAMPP_PAIRED_CONSISTENCY_LONG120_PREREG_2026-08-30.md`.  Its no-consistency
control, `p4-campp-known446-ood-channelrobust-paired-control-long120-oof-f0`,
started first from the selected P3 Raw checkpoint on commit `e1059813`.  Both
control and treatment have a fixed 120-epoch cosine schedule, no metric early
stopping, and identical data/model/optimisation settings; the treatment's only
scientific difference is the fixed-weight clean/aug cosine term.  Epochs 40 and
80 are diagnostics only.

The first 16 complete long-control epochs reproduce the earlier 40-epoch
engineering control closely: mean Raw Macro-F1 is `0.93793125` versus
`0.9380944840` (delta `-0.0001632340`), and both early bests are approximately
`0.94110`.  This argues against catastrophic warm-start forgetting or a broken
long scheduler, but is deliberately non-decision evidence.  The control must
finish before the treatment starts, and the harder treatment receives the same
120-epoch opportunity.  A further extension is allowed only if the locked late
tail checks pass for the treatment relative to this compute-matched control;
the MeMo result above is explicit counter-evidence against assuming that an
embedding consistency term improves indefinitely with time.

### Historical P3 channel-robust source

The historical source profile was `p3-campp-known446-ood-channelrobust-oof-f0`,
commit `eb433629b85a07a0665056d1bf6fcd84694cf1ca`.  It changed only the augmentation
policy relative to Control Fold 0.  Its early epoch-8 advantage did not persist
through the frozen-encoder phase, but the post-unfreeze trajectory is converging
toward Control.  At matched epoch 33, treatment probability-average Macro-F1
was `0.9210` versus Control `0.9222834` (about `-0.0013`), while logit-average
was slightly higher (`0.9160` versus `0.9157215`).  Treatment EMA Macro-F1 was
`0.9106` versus Control `0.9073612` (about `+0.0032`), and training loss was
lower (`3.1711` versus `3.3714`); validation loss remained nearly equal but
slightly higher (`1.1514` versus `1.1499`).  This is evidence of continuing
adaptation, not a terminal win: Raw is still oscillatory, and neither selected
checkpoint nor LME20 complementarity is known.  The run remains healthy and
active, so checkpoint selection and downstream gates stay locked until terminal
evaluation.  The epoch-40 futility rule only rejects a best Raw score below
`0.90`; it is not a reason to stop this trajectory after that threshold has
already been exceeded.

The treatment still passes only if the preregistered standalone LME20,
fixed-50/50 complementarity, Known/OOD, rescue-rate, and provenance gates all
pass after the run.  Failure rejects the candidate and forbids automatic Fold
expansion.

The later matched-epoch observation strengthens the case for letting the
harder treatment converge.  At epoch 42, treatment probability-average
Macro-F1 reached a new best of `0.9335` versus Control `0.9282921` (about
`+0.0052`), and treatment logit-average was `0.9277` versus Control
`0.9223749` (about `+0.0053`).  Training loss was lower (`2.4337` versus
`2.5685`) while validation loss remained nearly matched (`1.1946` versus
`1.1920`).  EMA still lagged (`0.9225` versus `0.9268546`), which is consistent
with a slow shadow-model response but prevents calling epoch 42 a broad or
terminal win.  Agreement of both Raw aggregation rules is nevertheless a
material signal that the channel-robust representation may now be overtaking
Control; the selected checkpoint, LME20 and complementarity gates remain
locked until the source run is terminal.

Additional time subsequently produced a genuine new Raw best at epoch 56:
probability-average Macro-F1 `0.9338324913327573`, exceeding the epoch-42 best
and resetting patience. The exact Raw checkpoint stores optimizer state and a
scheduler state at epoch 56, and MLflow contains 56 contiguous core-metric
points. Training loss had fallen to `1.9042` while validation loss was
`1.2355`; this is continued adaptation with a modest generalisation gap, not
proof of terminal superiority. The new best validates the decision not to stop
on short-term oscillation, while all LME20/fusion/guardrail gates remain
unchanged and deferred until terminal evaluation.

By epoch 66, Raw remained oscillatory (`0.9295613`) but EMA had climbed to
`0.9331220`, only about `0.00352` below matched Control EMA.  The locked
non-terminal continuation preview at epoch 65 had already passed two of three
trend tests: final-ten EMA slope `+0.0003928554` per epoch and loss adaptation
with a `0.1051542` training-loss decrease against only `0.0115912` validation-
loss increase.  Epoch 66 therefore strengthens the interpretation of slow
adaptation without changing the fact that Raw, not EMA, selects the terminal
model and that downstream evidence remains unavailable until the source stops.

Additional wall time again mattered: epoch 81 established a later Raw
probability-average best of `0.9398682068`, only `0.0070529838` below Control's
selected Fold-0 Raw score `0.9469211906`.  The source remained healthy through
epoch 94 with no NaN/OOM and 94 contiguous MLflow points.  Raw at epoch 94 was
`0.9336545320`, EMA `0.9351693793`, training loss `1.4600201` and validation
loss `1.2723910`.  The late best justifies allowing the locked patience/six-hour
boundary to operate; the subsequent non-improvement also prevents an unlimited
epoch extension without the preregistered trend gate.

## Decision-selector oracle ceiling

The locked three-Fold residual audit contains `4447` OOF files and `131` LME20
errors.  Relative to the raw head, LME20 corrected `120` head errors while the
head was correct on only `14` LME20 errors.  Therefore an impossible oracle
that chooses perfectly between the head and LME20 can save at most 14 additional
files, an accuracy ceiling of `14 / 4447 = 0.00314819`; this is not a guaranteed
Macro-F1 gain.  A learned selector would also need to preserve the 120 LME20-only
rescues.  The error-feature distributions overlap: mean fused confidence is
`0.4372` for Known-to-Unknown and `0.4302` for Unknown-to-Known errors.

This evidence deprioritises another generic head/LME selector or global
threshold sweep.  The remaining high-value hypothesis class is representation
or condition modelling that changes the residual topology itself.  Any later
selector must first demonstrate cross-fit separability and must remain subject
to simultaneous Known/OOD guardrails.

## Prototype-aggregation cross-fit update

The completed leave-one-fold-out aggregation audit tested 11 train-only
prototype aggregation rules.  The best fixed family under per-target-fold
selection was `logmeanexp_b20`: aggregate Macro-F1 `0.9629589450`, minimum
held-out Fold gain `+0.0136101890`, aggregate Known Accuracy delta
`+0.0017953321`, and OOD-F1 delta `+0.0200328352`.  This independently confirms
that log-mean-exp prototype evidence is stable across all three Folds.

It does **not** justify a new submission: the locked operational LME20/PCM OOF
score is `0.9633564052`, so the new cross-fit policy is lower by about
`0.00039746`.  The leaderboard is not used to choose another aggregation or
decision parameter.  Generic aggregation sweeps are therefore deprioritised;
their scientific value is confirmation of the LME mechanism, not a replacement
for the current record-setting package.

## Backup decision order

1. Finish the active channel-robust Fold-0 source without intervention.
2. If its locked terminal trend gate passes, run the preregistered stateful
   continuation once, within three hours and `$0.55`; never run it in parallel.
3. Audit the terminal source/continuation with standalone LME20, fixed 50/50
   complementarity, rescue rate, Known/OOD and provenance gates.
4. Only a gate-passing candidate may receive a separate Fold-1/2
   preregistration.  A failure triggers residual-topology diagnosis, not a tuned
   fusion weight.
5. If failure is compatible with augmentation residual, run the already
   preregistered paired clean/aug A/B experiment.  If it instead indicates that
   fixed-hard exposure itself is the problem, preregister one fixed curriculum
   schedule with breakpoints chosen from literature rather than Fold 0.
6. Condition-aware prototype banks and domain-adversarial training remain later
   options only after these cheaper, more isolated hypotheses are resolved.

## SciSpace retrieval update (2026-08-30)

A semantic SciSpace review was run specifically for cross-recording
same-speaker objectives, channel/session mismatch, hard sampling, and the
question of whether an auxiliary contrastive objective should remain active
through long fine-tuning.  The retrieval produced three useful boundaries for
the active paired experiment:

- SSPS (Interspeech 2025, DOI `10.21437/Interspeech.2025-183`) is direct
  evidence that a positive from the same speaker but a different recording
  condition can reduce intra-speaker variance; the paper reports benefit in
  both SimCLR and DINO.  This supports the *identity of the positive pair* in
  the preregistered treatment, not its fixed weight or duration.
- Asymmetric clean-segment guidance (ICASSP 2024, DOI
  `10.1109/ICASSP48485.2024.10446161`) reports a relative improvement from
  explicitly pairing clean and augmented segments.  It also stresses that
  augmentation must preserve speaker information.  This supports the matched
  clean/aug construction and the embedding-spread guardrail; it does not make
  augmentation severity or loss weight transferable to this competition.
- Session-embedding compensation (ICASSP 2024, DOI
  `10.1109/ICASSP48485.2024.10445987`) shows that session information can be
  modeled as a separate compensating score while the speaker extractor stays
  fixed.  It is therefore a credible later condition-modeling branch if the
  current invariance treatment fails, but it should not be bundled into the
  active single-variable test.

SciSpace also retrieved work that keeps a self-supervised EMA-target loss
through fine-tuning (ICASSP 2024, DOI `10.1109/ICASSP48485.2024.10446468`) and
work using two-stage or joint supervised/contrastive training.  The available
abstract-level evidence does not provide a speaker-specific controlled
comparison of a constant auxiliary loss against a scheduled-off version at a
long horizon.  The earlier MeMo evidence remains the more relevant warning
that embedding-side benefits can concentrate early.  Consequently, the active
control/treatment contract is unchanged: finish both matched 120-epoch arms,
inspect late-tail gain and collapse guardrails, and permit a matched extension
only if every preregistered tail condition passes.  There is no literature
basis for extending only the treatment or changing its weight mid-run.

At the time of this update the active control had 20 complete epochs and 20/20
contiguous MLflow points for 33 metrics.  Its best observed Raw probability-
average Macro-F1 remained approximately `0.9411` (epochs 13--14); epoch 20 was
`0.9356094`, with EMA `0.9386031`, Known Accuracy `0.9495516`, and OOD-F1
`0.9394773`.  These are non-terminal monitoring observations and do not select
a checkpoint or alter the paired gate.

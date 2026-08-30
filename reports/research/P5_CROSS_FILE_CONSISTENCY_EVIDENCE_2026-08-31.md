# P5 Cross-File Consistency: Evidence and Transfer Limits (2026-08-31)

## Decision context

P5 tests one locked change against a matched 120-epoch CAM++ control: cosine consistency with weight `0.1` between different files assigned to the same speaker. The current treatment must not be reconfigured mid-run. This note records what the literature does and does not justify so the terminal interpretation is not retrofitted to the observed score.

## Direct evidence

Lin et al. (ICASSP 2020) combine frame-level and utterance-level maximum mean discrepancy with consistency regularization for unlabeled target-domain speech. Their consistency pair is a target utterance and an augmented version of that same utterance (noise or reverberation), and the embedding discrepancy is minimized between the clean and perturbed views. The study reports that consistency helps in NIST SRE16/SRE18 and that the full front-end adaptation complements backend adaptation.

This supports the general proposition that embedding consistency can improve robustness to domain perturbations. It does **not** directly validate forcing embeddings from two naturally different recordings of a speaker to agree. Cross-file pairs contain additional phonetic, duration, affect, session, channel, and quality differences, so their consistency loss is a materially stronger invariance constraint.

Li et al. (ICASSP 2025) likewise align noisy-clean views and combine feature-level alignment, supervised contrastive separation, and a redundancy-reduction objective. The abstract-level evidence again concerns controlled noisy-clean views, not arbitrary cross-session recordings.

Wang et al. (ICASSP 2023) show that recording-dependent embedding uncertainty can matter substantially under domain mismatch. Their gains come from propagating uncertainty through a PLDA backend, not from collapsing uncertain and reliable recordings to the same point. This is relevant to P5 because equal weighting of all cross-file pairs may over-regularize low-quality or short recordings.

## Locked interpretation rules for P5

1. P5 is an empirical extrapolation beyond the closest direct literature, not a paper replication.
2. An isolated best epoch is insufficient. Accept only the preregistered paired terminal gate using the selected-treatment LME20, matched control, Known/OOD guardrails, rescue rate, spread ratio, and artifact audit.
3. Do not increase the consistency coefficient or broaden unfreezing in response to a weak P5 trajectory.
4. If P5 is neutral or negative, the evidence-supported successor is a separately preregistered same-utterance perturbation or uncertainty/quality-weighted consistency ablation—not an untuned repetition of cross-file consistency.
5. If P5 passes, the pass establishes dataset-specific evidence for this stronger constraint and permits multifold preregistration; it does not by itself authorize a submission.

## Sources

- Weiwei Lin et al., “Multi-Level Deep Neural Network Adaptation for Speaker Verification Using MMD and Consistency Regularization,” ICASSP 2020, DOI: [10.1109/ICASSP40776.2020.9054134](https://doi.org/10.1109/ICASSP40776.2020.9054134). Author-hosted presentation: [3043-LinMakLiSuYu.pdf](https://sigport.org/sites/default/files/docs/3043-LinMakLiSuYu.pdf).
- Zuoliang Li et al., “Aligning Noisy-Clean Speech Pairs at Feature and Embedding Levels for Learning Noise-Invariant Speaker Representations,” ICASSP 2025, DOI: [10.1109/ICASSP49660.2025.10889792](https://doi.org/10.1109/ICASSP49660.2025.10889792).
- Qiongqiong Wang, Kong Aik Lee, and Tianchi Liu, “Incorporating Uncertainty from Speaker Embedding Estimation to Speaker Verification,” ICASSP 2023 / [arXiv:2302.11763](https://arxiv.org/abs/2302.11763).

## Retrieval note

SciSpace semantic search identified the relevant studies. Its requested methodology/result/limitation columns returned no paper-level data for the selected records, so the transfer analysis above was checked against the author/IEEE presentation, DOI metadata, and the primary arXiv full text rather than inferred from SciSpace snippets alone.

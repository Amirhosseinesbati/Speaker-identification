# P5 Session/Channel Metadata Audit — 2026-08-30

## Purpose

Determine whether the competition files contain a direct recording-session or
channel identifier that could make the P5 positive sampler explicitly
cross-session rather than only cross-file.  This is a read-only EDA audit; it
does not modify the locked P5 control/treatment pair.

## Manifest evidence

The authoritative raw manifest `data/raw/labels.csv` contains only
`speaker_id` and `audio_file`.  The processed manifests add competition and
training labels (`original_speaker_id`, `is_ood`, `metric_label`, and `label`)
but no session, channel, device, source, or recording identifier.

All inspected audio paths are flat basenames.  Their stems follow UUID form,
for example five random hexadecimal groups separated by hyphens.  The UUID is
therefore a file identifier and must not be reinterpreted as a session or
channel grouping.

## Container/codec evidence

A deterministic stride sample of 40 rows from the raw manifest was inspected
with `ffprobe` under low CPU priority.  One sampled file was unreadable.  Every
one of the 39 readable files had the same observed profile despite its `.mp3`
suffix:

- container reported as WAV;
- codec `pcm_s16le`;
- sample rate `16000` Hz;
- two channels;
- nominal bitrate `512000` bit/s;
- no encoder tag.

This small sample is sufficient to reject encoder/container tags as a useful
direct channel label.  It is not a claim that all acoustic recording
conditions are identical: microphone, room, codec history before export, and
speaker session can still differ inside the uniform PCM representation.

## Scientific consequence

P5 guarantees two distinct files from the same authoritative known speaker;
it does **not** guarantee two known sessions or devices.  Reports and decisions
must therefore call it `cross-file`, not `cross-session`.

The lack of direct channel labels also keeps channel-adversarial training out
of scope: constructing a target from UUIDs, file size, duration, RMS, or other
quality proxies would conflate recording condition with content and would not
be a clean single-variable ablation.  Duration/RMS/activity/window features
remain diagnostic covariates only.  The locked P5 pair remains valid as a test
of distinct-file compactness, with this interpretation limit recorded before
its terminal treatment result.


# Vast.ai Instance Acceptance — 48886926

Date: 2026-08-27

## Contract

- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- Host CPU: AMD Ryzen 9 5950X
- Effective vCPUs: 32
- RAM: 35.15 GiB
- Storage: 100 GiB
- Location: Sweden
- On-demand total: $0.171111/hour, including 100 GiB storage
- Host driver: 595.71.05; maximum supported CUDA: 13.2
- Container: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`

## Measured acceptance results

- FP16 matrix multiplication median: 77.87 TFLOPS
- FP16 matrix multiplication minimum: 77.06 TFLOPS
- Host-to-device transfer: 13.03 GB/s
- Sequential synchronized disk write: 1.9 GB/s
- Sequential direct disk read: 3.4 GB/s
- External download, 100 MB from Google Cloud: 26.24 MB/s (about 210 Mbps)
- CPU multiprocessing: 16/16 workers completed the four-second workload; 5.47 s wall time including process startup
- Effective filesystem capacity: 100 GiB

## Decision

Accepted for campaign bootstrap. GPU, PCIe, CPU, memory and disk satisfy the
training-worker gate. Measured external download is below the marketplace claim
of 820 Mbps, but remains sufficient for the expected dataset and checkpoint
traffic. Upload throughput will also be measured during the real workspace copy.

## Open issue

Resolved. The bot credential and allowlisted private chat id are stored only in
worker-side files with mode `0600`; no token value is stored in Git, command
arguments or this report. A live Persian notification test succeeded.

## Re-validation after campaign checkout

The benchmark was repeated from the campaign Python environment after checkout:

- PyTorch `2.13.0+cu130`; CUDA available; compute capability 8.6.
- FP16 matrix multiplication median/minimum: 77.50 / 76.79 TFLOPS.
- Host-to-device transfer: 12.58 GB/s.
- External 100 MB download: 216.88 Mbps.
- 16/16 CPU workers completed; 5.79 s wall time.
- 23.56 GiB usable VRAM and 50 GiB free disk after deterministic WAV conversion.

The small variance from the first acceptance run is normal. Both runs pass the
worker gate with ample margin.

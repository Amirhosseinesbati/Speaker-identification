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

`TELEGRAM_BOT_TOKEN` exists in the Vast account secret store but was not injected
into this third-party container. It must be applied as a per-instance environment
variable before enabling the notifier. No token value is stored in this report.

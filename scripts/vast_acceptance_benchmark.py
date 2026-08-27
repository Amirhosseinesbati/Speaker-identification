"""Acceptance benchmark for a freshly rented Vast.ai training worker.

The script is dependency-light beyond PyTorch and emits a single JSON document so
the campaign supervisor can gate a host before uploading the competition data.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import statistics
import time
import urllib.request

import torch


def _cpu_worker(seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    value = 0x12345678
    iterations = 0
    while time.perf_counter() < deadline:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        iterations += 1
    return iterations ^ value


def cpu_parallel_benchmark(workers: int, seconds: float = 4.0) -> dict:
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_cpu_worker, [seconds] * workers)
    return {
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "completed_workers": len(results),
    }


def gpu_benchmark() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    size = 8192
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()

    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
        samples.append((2.0 * size**3) / elapsed_s / 1e12)

    host = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(8):
        host.to(device, non_blocking=True)
    torch.cuda.synchronize()
    h2d_gbps = (host.numel() * host.element_size() * 8) / (time.perf_counter() - started) / 1e9

    return {
        "available": True,
        "name": props.name,
        "vram_gib": props.total_memory / 2**30,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "compute_capability": f"{props.major}.{props.minor}",
        "fp16_matmul_tflops_median": statistics.median(samples),
        "fp16_matmul_tflops_min": min(samples),
        "h2d_gbps": h2d_gbps,
    }


def network_download_benchmark(bytes_to_download: int = 100_000_000) -> dict:
    url = "https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz"
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{bytes_to_download - 1}"},
    )
    started = time.perf_counter()
    received = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
    elapsed = time.perf_counter() - started
    return {
        "bytes": received,
        "seconds": elapsed,
        "mbps": received * 8 / elapsed / 1e6,
    }


def main() -> None:
    cpu_count = os.cpu_count() or 1
    report = {
        "platform": platform.platform(),
        "cpu_count": cpu_count,
        "cpu_parallel": cpu_parallel_benchmark(min(cpu_count, 16)),
        "gpu": gpu_benchmark(),
    }
    try:
        report["network_download"] = network_download_benchmark()
    except Exception as exc:  # Network failure is reported, not hidden.
        report["network_download"] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

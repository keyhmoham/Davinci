"""Measure Davinci prefill/decode latency and KV-cache memory from a checkpoint.

Example:
  PYTHONPATH=. python experiments/benchmark_context.py --checkpoint /path/to/checkpoint --lengths 4096,8192
This is a hardware-specific microbenchmark, not a quality benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from src.davinci.model import DavinciForCausalLM


@torch.inference_mode()
def benchmark_length(model, length: int, *, device: torch.device, decode_steps: int) -> dict:
    if length < 2 or decode_steps < 1:
        raise ValueError("length must be >= 2 and decode_steps must be positive")
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, length), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    logits, caches = model.prefill(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - start
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start = time.perf_counter()
    for step in range(decode_steps):
        logits = model.decode(next_id, length + step, caches)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - start
    cache_bytes = 0
    for cache in caches:
        for name in ("key", "value"):
            tensor = getattr(cache, name, None)
            if isinstance(tensor, torch.Tensor):
                cache_bytes += tensor.numel() * tensor.element_size()
    return {
        "sequence_tokens": length,
        "decode_steps": decode_steps,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": length / max(prefill_seconds, 1e-12),
        "decode_seconds_total": decode_seconds,
        "decode_tokens_per_second": decode_steps / max(decode_seconds, 1e-12),
        "kv_cache_bytes": cache_bytes,
        "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lengths", default="1024,2048,4096,8192")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    try:
        lengths = [int(item) for item in args.lengths.split(",")]
    except ValueError:
        parser.error("--lengths must be comma-separated integers")
    if not lengths or any(length < 2 for length in lengths) or args.decode_steps < 1:
        parser.error("lengths must be >= 2 and decode-steps must be positive")
    model = DavinciForCausalLM.from_pretrained(args.checkpoint, map_location=device).to(device).eval()
    results = []
    for length in lengths:
        try:
            results.append(benchmark_length(model, length, device=device, decode_steps=args.decode_steps))
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append({"sequence_tokens": length, "error": "cuda_out_of_memory"})
            break
    report = {
        "warning": "Random-token performance microbenchmark only; not model-quality or long-context capability evidence.",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

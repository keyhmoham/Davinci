"""Atomic, reproducible training checkpoints for trusted local runs.

Checkpoint files use PyTorch serialization and must only be loaded from trusted
sources. They contain optimizer state and Python/NumPy/PyTorch RNG state.
"""
from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_FORMAT_VERSION = 1


def _model_config(model) -> dict[str, Any] | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return None


def _capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError as exc:
            raise RuntimeError("Checkpoint contains NumPy RNG state but NumPy is unavailable") from exc
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from the checkpoint RNG state")
        torch.cuda.set_rng_state_all(cuda_states)


def _gradient_state(model) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _restore_gradients(model, gradients: Mapping[str, torch.Tensor | None]) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(gradients):
        raise ValueError("Checkpoint gradient parameter names do not match the model")
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if gradient is None:
            parameter.grad = None
        else:
            if gradient.shape != parameter.shape:
                raise ValueError(f"Gradient shape mismatch for parameter {name}")
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype).clone()


def save_training_checkpoint(
    path: str | os.PathLike[str],
    model,
    optimizer,
    *,
    step: int,
    micro_step: int = 0,
    scheduler=None,
    scaler=None,
    metadata: Mapping[str, Any] | None = None,
    data_state: Mapping[str, Any] | None = None,
    save_gradients: bool = True,
) -> Path:
    """Atomically save complete training state; step counts completed optimizer steps.

    data_state should include a deterministic sampler/stream cursor (for example
    epoch, record offset, and shuffle-buffer state). Exact replay depends on the
    caller restoring that data pipeline consistently.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if isinstance(micro_step, bool) or not isinstance(micro_step, int) or micro_step < 0:
        raise ValueError("micro_step must be a non-negative integer")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "micro_step": micro_step,
        "model_config": _model_config(model),
        "metadata": dict(metadata or {}),
        "data_state": dict(data_state or {}),
        "rng_state": _capture_rng_state(),
        "gradients": _gradient_state(model) if save_gradients else None,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        # Persist the directory entry where supported.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_training_checkpoint(
    path: str | os.PathLike[str],
    model,
    optimizer,
    *,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    expected_metadata: Mapping[str, Any] | None = None,
    restore_rng: bool = True,
    restore_gradients: bool = True,
) -> dict[str, Any]:
    """Restore a trusted checkpoint and return step/data/metadata information.

    The file is loaded with weights_only=False because optimizer and RNG state
    include non-tensor Python objects. Never load a checkpoint from an untrusted
    source. Metadata keys supplied in expected_metadata must match exactly.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported or malformed training checkpoint")
    expected_config = _model_config(model)
    if payload.get("model_config") != expected_config:
        raise ValueError("Checkpoint model configuration does not match the target model")
    saved_metadata = payload.get("metadata", {})
    for key, expected in (expected_metadata or {}).items():
        if saved_metadata.get(key) != expected:
            raise ValueError(f"Checkpoint metadata mismatch for {key!r}")
    if (scheduler is None) != (payload.get("scheduler") is None):
        raise ValueError("Scheduler presence does not match the checkpoint")
    if (scaler is None) != (payload.get("scaler") is None):
        raise ValueError("GradScaler presence does not match the checkpoint")

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_gradients and payload.get("gradients") is not None:
        _restore_gradients(model, payload["gradients"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return {
        "step": int(payload["step"]),
        "micro_step": int(payload["micro_step"]),
        "metadata": dict(saved_metadata),
        "data_state": dict(payload.get("data_state", {})),
        "model_config": payload.get("model_config"),
    }

"""Model runtime: device/dtype resolution, resident loader, memory probes.

Binding constraints (docs/m1-design.md §0, spike REPORT.md §4/§5):

* ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`` must be set BEFORE torch is
  imported, otherwise the Metal driver cache accumulates ~20 GB/step sampling
  transients and OOMs. This module sets it at import time, before ``import
  torch`` — keep it that way (temper/model/__init__.py enforces the order).
* ``model.config.use_cache = False`` immediately after load: transformers
  >=4.47 otherwise auto-builds a DynamicCache inside the diffusion loop and
  pollutes attention across denoise steps -> degenerate text (spike blocker
  #2, quality blocker #1).
* MPS + bf16 is the production path; CPU exists only as a loud debugging
  fallback.
* The HF cache on disk is never modified. No monkeypatching is installed:
  the spike measured K=1 sequential canvases clean with just the two fixes
  above (NaN logits only affected padded batch rows, which K=1 never has).

Stdlib + venv deps only. Python 3.9 compatible.
"""

from __future__ import annotations

import os
import sys

# ---- MUST run before `import torch` ---------------------------------------
# 0.0 disables the high-watermark "limit" that lets the Metal driver cache
# grow without bound between empty_cache() calls (spike §4: reproducible OOM
# at K>=4 without it; mandatory even for long K=1 sequences).
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch  # noqa: E402  (import AFTER the env fix above)

from transformers import AutoModel, AutoTokenizer  # noqa: E402

__all__ = ["MODEL_ID", "resolve_device", "get_model", "mps_memory_snapshot"]

MODEL_ID = "Dream-org/Dream-Coder-v0-Instruct-7B"

# Venv pins this milestone was validated against (docs/m1-design.md §0):
# torch==2.8.0, transformers==4.57.6, accelerate==1.10.1, huggingface_hub==0.36.2

_RESIDENT: dict = {"tokenizer": None, "model": None}


def resolve_device():
    """Return (device_str, dtype). mps + bf16 is required for real runs.

    The CPU path exists for debugging only and warns loudly: Dream's
    diffusion loop on CPU is 10-50x slower and some bf16 kernels degrade.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", torch.bfloat16
    msg = (
        "temper.model: MPS is NOT available — falling back to CPU. This is a "
        "DEBUGGING path only: expect 10-50x slowdown per denoise step and "
        "verify results before trusting anything. Real runs require "
        "macOS arm64 + Metal (torch.backends.mps)."
    )
    print("!!! " + msg + " !!!", file=sys.stderr)
    import warnings

    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return "cpu", torch.bfloat16


def _load_once():
    device, dtype = resolve_device()
    t0 = __import__("time").perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,            # transformers 4.57 arg name (was torch_dtype)
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    # MANDATORY (spike blocker #2): transformers >=4.47 otherwise builds a
    # DynamicCache inside the diffusion loop and pollutes attention across
    # denoise steps -> degenerate/repetitive output.
    model.config.use_cache = False
    load_s = __import__("time").perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"temper.model: loaded {MODEL_ID} "
        f"({n_params/1e9:.2f}B params, {str(dtype).split('.')[-1]}, "
        f"device={device}, use_cache={model.config.use_cache}) "
        f"in {load_s:.1f}s",
        flush=True,
    )
    return tokenizer, model


def get_model():
    """Resident holder: load tokenizer+model once per process, reuse after."""
    if _RESIDENT["model"] is None:
        _RESIDENT["tokenizer"], _RESIDENT["model"] = _load_once()
    return _RESIDENT["tokenizer"], _RESIDENT["model"]


def mps_memory_snapshot():
    """Best-effort memory counters (MB); torch.mps fields are 0/None on CPU."""
    snap = {
        "mps_allocated_MB": None,
        "mps_driver_MB": None,
        "mps_recommended_max_MB": None,
        "rss_peak_MB": None,
    }
    try:
        if torch.backends.mps.is_available():
            snap["mps_allocated_MB"] = torch.mps.current_allocated_memory() / 2**20
            snap["mps_driver_MB"] = torch.mps.driver_allocated_memory() / 2**20
            snap["mps_recommended_max_MB"] = (
                torch.mps.recommended_max_memory() / 2**20
            )
    except Exception:
        pass
    try:
        import resource

        # macOS: ru_maxrss is in bytes.
        snap["rss_peak_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        pass
    return snap

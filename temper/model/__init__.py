"""temper.model — Dream-Coder-v0-Instruct-7B diffusion runtime (M1).

Public surface:
    from temper.model import get_model, sample, extract_rust

``temper.model.runtime`` MUST be imported before ``torch`` anywhere in the
process (it sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 pre-import); importing
this package first guarantees that ordering.
"""

from temper.model.runtime import get_model, resolve_device, mps_memory_snapshot
from temper.model.dream import sample, ParticleResult
from temper.model.extract import extract_rust, ExtractResult

__all__ = [
    "get_model",
    "resolve_device",
    "mps_memory_snapshot",
    "sample",
    "ParticleResult",
    "extract_rust",
    "ExtractResult",
]

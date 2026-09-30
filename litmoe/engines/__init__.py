"""Engine adapters."""
from .base import Engine
from .dwarfstar import DwarfStarEngine, is_installed as dwarfstar_installed
from .ktransformers import KtransformersEngine, is_installed as kt_installed
from .llamacpp import LlamaCppEngine, is_installed as llama_installed
from .warp import WarpEngine, is_installed as warp_installed


def make_engine(model):
    """Factory: build the right engine for a model entry."""
    if model.engine == "ktransformers":
        return KtransformersEngine(model)
    elif model.engine == "llamacpp":
        return LlamaCppEngine(model)
    elif model.engine == "warp":
        return WarpEngine(model)
    elif model.engine == "dwarfstar":
        return DwarfStarEngine(model)
    else:
        raise ValueError(f"Unknown engine: {model.engine}")


__all__ = [
    "Engine", "KtransformersEngine", "LlamaCppEngine", "WarpEngine",
    "DwarfStarEngine", "make_engine", "kt_installed", "llama_installed",
    "warp_installed", "dwarfstar_installed",
]

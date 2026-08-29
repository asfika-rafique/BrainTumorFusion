"""Dataset discovery, split manifests, and PyTorch data loaders.

The public dataset symbols are imported lazily so the hash-based split tools
remain usable in a lightweight audit environment without PyTorch installed.
"""

__all__ = ["BrainTumorDataset", "make_loaders_from_cfg"]


def __getattr__(name: str):
    if name in __all__:
        from .datasets import BrainTumorDataset, make_loaders_from_cfg

        return {"BrainTumorDataset": BrainTumorDataset, "make_loaders_from_cfg": make_loaders_from_cfg}[name]
    raise AttributeError(name)

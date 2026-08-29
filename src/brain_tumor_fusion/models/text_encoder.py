"""Text encoder extension point.

The original project included a placeholder identity module rather than a
trained language encoder. It is intentionally not used by active configs.
"""

from __future__ import annotations

import torch.nn as nn


def build_text_encoder(*_args, **_kwargs) -> nn.Module:
    """Raise clearly instead of silently pretending text fusion is active."""

    raise NotImplementedError(
        "No trained text encoder is implemented in this repository; use an image-only config."
    )

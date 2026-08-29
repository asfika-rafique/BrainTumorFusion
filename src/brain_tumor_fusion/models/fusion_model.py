"""Image/text fusion classifier used by the research experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    """Project encoder features into a shared classifier head.

    The active repository configuration uses the image branch. The text branch
    remains an explicit extension point, but is not enabled until a real text
    encoder and an auditable caption protocol are added.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        img_out_dim: int | None = None,
        text_encoder_name: str = "bert-base-uncased",
        txt_out_dim: int = 768,
        fusion_hidden: int = 512,
        num_classes: int = 4,
        use_text: bool = False,
        dropout: float = 0.3,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.use_text = use_text
        self.text_encoder = text_encoder
        self.text_encoder_name = text_encoder_name

        image_dim = getattr(image_encoder, "out_dim", None) or img_out_dim
        if image_dim is None:
            image_dim = self._infer_image_dim(image_encoder)
        self.actual_img_dim = int(image_dim)

        self.img_proj = nn.Sequential(
            nn.Linear(self.actual_img_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if use_text:
            if text_encoder is None:
                raise ValueError("A text encoder is required when use_text=True")
            self.txt_proj = nn.Sequential(
                nn.Linear(txt_out_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fusion_input_dim = fusion_hidden * 2
        else:
            fusion_input_dim = fusion_hidden

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_hidden // 2, num_classes)
        self._initialize_weights()

    @staticmethod
    def _infer_image_dim(encoder: nn.Module) -> int:
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            features = encoder(torch.zeros(1, 3, 224, 224))
        encoder.train(was_training)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return int(features.shape[1])

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.image_encoder(image)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        image_hidden = self.img_proj(features)

        if self.use_text:
            if input_ids is None or attention_mask is None or self.text_encoder is None:
                raise ValueError("Text inputs and a text encoder are required for text fusion")
            text_hidden = self.text_encoder(input_ids, attention_mask)
            if isinstance(text_hidden, tuple):
                text_hidden = text_hidden[0]
            text_hidden = self.txt_proj(text_hidden)
            features = torch.cat([image_hidden, text_hidden], dim=-1)
        else:
            features = image_hidden

        return self.head(self.fusion(features))

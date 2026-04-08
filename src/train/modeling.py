import torch
import warnings
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BuildImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, feature_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return x.flatten(1)


class BuildAttentionMILModel(nn.Module):
    """Encode each image with a CNN backbone, then aggregate per-build with attention."""

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = BuildImageEncoder(feature_dim=256)
        in_features = self.encoder.feature_dim

        attn_hidden = max(in_features // 2, 128)
        self.attention = nn.Sequential(
            nn.Linear(in_features, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if images.dim() != 5:
            raise ValueError(f"Expected input shape [B, N, C, H, W], got {tuple(images.shape)}")

        batch_size, num_images, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_images, channels, height, width)
        encoded = self.encoder(flat_images)
        encoded = encoded.view(batch_size, num_images, -1)

        if mask is None:
            mask = torch.ones((batch_size, num_images), device=images.device, dtype=torch.bool)
        else:
            mask = mask.to(device=images.device, dtype=torch.bool)

        attn_scores = self.attention(encoded).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights * mask.float()
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


def make_model(num_classes: int) -> nn.Module:
    return BuildAttentionMILModel(num_classes=num_classes)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA device requested, but torch.cuda.is_available() is False. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device(device_arg)

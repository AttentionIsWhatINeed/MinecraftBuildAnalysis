import torch
import warnings
from torch import nn
from torchvision import models


def _build_pretrained_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feature_dim

    if backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feature_dim

    raise ValueError(
        f"Unsupported backbone_name='{backbone_name}'. Supported: resnet18, efficientnet_b0"
    )


class PretrainedBuildImageEncoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        try:
            backbone, feature_dim = _build_pretrained_backbone(backbone_name, pretrained)
        except Exception as exc:
            if not pretrained:
                raise

            warnings.warn(
                f"Failed to load pretrained weights for '{backbone_name}': {exc}. "
                "Falling back to randomly initialized backbone.",
                RuntimeWarning,
                stacklevel=2,
            )
            backbone, feature_dim = _build_pretrained_backbone(backbone_name, pretrained=False)

        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable


class MILModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.2,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = True,
        class_specific_attention: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_specific_attention = bool(class_specific_attention)
        self.encoder = PretrainedBuildImageEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        in_features = self.encoder.feature_dim

        attn_hidden = max(in_features // 2, 128)
        attn_out_dim = self.num_classes if self.class_specific_attention else 1
        self.attention = nn.Sequential(
            nn.Linear(in_features, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, attn_out_dim),
        )

        if self.class_specific_attention:
            self.classifier_norm = nn.LayerNorm(in_features)
            self.classifier_dropout = nn.Dropout(dropout)
            self.classifier_weight = nn.Parameter(torch.empty(self.num_classes, in_features))
            self.classifier_bias = nn.Parameter(torch.zeros(self.num_classes))
            nn.init.xavier_uniform_(self.classifier_weight)
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, self.num_classes),
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

        attn_scores = self.attention(encoded)

        if self.class_specific_attention:
            mask_3d = mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(~mask_3d, -1e4)
            attn_weights = torch.softmax(attn_scores, dim=1)
            attn_weights = attn_weights * mask_3d.float()
            attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

            pooled = torch.einsum("bnc,bnd->bcd", attn_weights, encoded)
            pooled = self.classifier_norm(pooled)
            pooled = self.classifier_dropout(pooled)
            logits = torch.einsum("bcd,cd->bc", pooled, self.classifier_weight)
            return logits + self.classifier_bias

        attn_scores = attn_scores.squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights * mask.float()
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.encoder.set_trainable(trainable)


def make_model(
    num_classes: int,
    dropout: float = 0.2,
    backbone_name: str = "resnet18",
    pretrained_backbone: bool = True,
    class_specific_attention: bool = False,
) -> nn.Module:
    return MILModel(
        num_classes=num_classes,
        dropout=dropout,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        class_specific_attention=class_specific_attention,
    )


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

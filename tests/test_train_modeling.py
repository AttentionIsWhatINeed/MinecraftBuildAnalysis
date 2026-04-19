import warnings

import pytest
import torch
from torch import nn

import src.train.modeling as modeling


class _DummyEncoder(nn.Module):
    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = 4
        self.backbone = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.backbone(flat)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def test_build_pretrained_backbone_invalid_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backbone_name"):
        modeling._build_pretrained_backbone("unknown", pretrained=False)


def test_pretrained_encoder_falls_back_when_pretrained_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_build(name: str, pretrained: bool):
        calls.append((name, pretrained))
        if pretrained:
            raise RuntimeError("no pretrained")
        return nn.Identity(), 3

    monkeypatch.setattr(modeling, "_build_pretrained_backbone", fake_build)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        encoder = modeling.PretrainedBuildImageEncoder(backbone_name="resnet18")

    assert calls == [("resnet18", True), ("resnet18", False)]
    assert encoder.feature_dim == 3
    assert any("Falling back to randomly initialized backbone" in str(item.message) for item in w)


def test_milmodel_forward_shape_masking_and_trainable_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling, "PretrainedBuildImageEncoder", _DummyEncoder)
    model = modeling.MILModel(num_classes=3, dropout=0.0, backbone_name="resnet18")

    images = torch.randn(2, 4, 3, 8, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    logits = model(images, mask)
    assert logits.shape == (2, 3)

    logits_nomask = model(images)
    assert logits_nomask.shape == (2, 3)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 3, 8, 8))

    model.set_backbone_trainable(False)
    assert all(not p.requires_grad for p in model.encoder.backbone.parameters())
    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.encoder.backbone.parameters())


def test_make_model_and_resolve_device_behaviors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling, "PretrainedBuildImageEncoder", _DummyEncoder)
    made = modeling.make_model(num_classes=2, dropout=0.1, backbone_name="resnet18")
    assert isinstance(made, modeling.MILModel)

    auto_device = modeling.resolve_device("auto")
    assert str(auto_device) in {"cpu", "cuda"}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cuda_device = modeling.resolve_device("cuda:0")
    assert str(cuda_device) == "cpu"
    assert any("falling back to CPU".lower() in str(item.message).lower() for item in w)

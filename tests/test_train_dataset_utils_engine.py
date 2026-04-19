import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.train.dataset import (
    MinecraftBuildBagDataset,
    _expand_to_build_samples,
    _load_json_builds,
    _resolve_image_path,
    build_datasets,
)
from src.train.engine import evaluate_f1_micro, run_epoch
from src.train.utils import set_seed


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_load_and_resolve_path_helpers(tmp_path: Path) -> None:
    list_file = tmp_path / "list.json"
    dict_file = tmp_path / "dict.json"
    bad_file = tmp_path / "bad.json"

    _write_json(list_file, [{"title": "a"}])
    _write_json(dict_file, {"builds": [{"title": "b"}]})
    _write_json(bad_file, {"unexpected": 1})

    assert _load_json_builds(list_file) == [{"title": "a"}]
    assert _load_json_builds(dict_file) == [{"title": "b"}]
    with pytest.raises(ValueError):
        _load_json_builds(bad_file)

    root = tmp_path / "root"
    assert _resolve_image_path("folder\\img.png", root) == root / "folder/img.png"


def test_expand_samples_and_dataset_bundle_building(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    _make_image(img1)
    _make_image(img2)

    builds = [
        {"title": "ok", "build_url": "u1", "tags": ["a", "x"], "local_image_paths": [str(img1), str(img2)]},
        {"title": "no_valid_tags", "build_url": "u2", "tags": ["z"], "local_image_paths": [str(img1)]},
        {"title": "missing_images", "build_url": "u3", "tags": ["a"], "local_image_paths": [str(tmp_path / "missing.png")]},
    ]
    tag_to_idx = {"a": 0, "x": 1}
    samples = _expand_to_build_samples(builds, tag_to_idx, tmp_path)
    assert len(samples) == 1
    assert samples[0]["title"] == "ok"
    assert samples[0]["target"].tolist() == [1.0, 1.0]
    assert len(samples[0]["image_paths"]) == 2

    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    test_json = tmp_path / "test.json"
    _write_json(train_json, {"builds": [{"title": "t", "tags": ["b", "a"], "local_image_paths": [str(img1)]}]})
    _write_json(val_json, {"builds": [{"title": "v", "tags": ["a"], "local_image_paths": [str(img2)]}]})
    _write_json(test_json, {"builds": [{"title": "te", "tags": ["b"], "local_image_paths": [str(img1)]}]})

    bundle = build_datasets(train_json, val_json, test_json, tmp_path)
    assert bundle.idx_to_tag == ["a", "b"]
    assert bundle.tag_to_idx == {"a": 0, "b": 1}
    assert len(bundle.train_build_samples) == 1
    assert len(bundle.val_build_samples) == 1
    assert len(bundle.test_build_samples) == 1

    _write_json(train_json, {"builds": [{"title": "empty", "tags": [], "local_image_paths": [str(img1)]}]})
    with pytest.raises(ValueError, match="No tags found in training set"):
        build_datasets(train_json, val_json, test_json, tmp_path)


def test_minecraft_bag_dataset_selection_and_item_shape(tmp_path: Path) -> None:
    img_paths = [tmp_path / f"img{i}.png" for i in range(3)]
    for path in img_paths:
        _make_image(path)

    sample = {
        "title": "s",
        "build_url": "u",
        "image_paths": img_paths,
        "target": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "true_tags": ["a"],
    }
    tf = transforms.ToTensor()

    with pytest.raises(ValueError):
        MinecraftBuildBagDataset([sample], tf, max_images_per_build=0, train_mode=False)

    eval_ds = MinecraftBuildBagDataset([sample], tf, max_images_per_build=2, train_mode=False, return_index=True)
    bag, mask, target, idx = eval_ds[0]
    assert bag.shape == (2, 3, 8, 8)
    assert mask.tolist() == [True, True]
    assert target.tolist() == [1.0, 0.0]
    assert idx == 0

    train_ds = MinecraftBuildBagDataset([sample], tf, max_images_per_build=2, train_mode=True)
    torch.manual_seed(0)
    selected = train_ds._select_paths(img_paths)
    assert len(selected) == 2
    assert all(path in img_paths for path in selected)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = images[:, 0, 0, 0, 0].unsqueeze(-1)
        return self.linear(x)


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.logits[: images.size(0)]


def test_run_epoch_and_evaluate_micro_metrics() -> None:
    device = torch.device("cpu")
    images = torch.tensor([[[[[1.0]]]], [[[[2.0]]]]], dtype=torch.float32)
    mask = torch.ones((2, 1), dtype=torch.bool)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loader = DataLoader(TensorDataset(images, mask, targets), batch_size=2, shuffle=False)

    model = _TinyModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler(enabled=False)

    train_loss = run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=False)
    eval_loss = run_epoch(model, loader, criterion, optimizer=None, device=device, scaler=scaler, use_amp=False)
    assert train_loss >= 0.0
    assert eval_loss >= 0.0

    fixed_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)
    metrics = evaluate_f1_micro(
        _FixedLogitModel(fixed_logits),
        loader,
        device=device,
        threshold=0.5,
        use_amp=False,
    )
    assert metrics["precision_micro"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["recall_micro"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["f1_micro"] == pytest.approx(1.0, abs=1e-6)


def test_set_seed_controls_random_and_torch() -> None:
    set_seed(123)
    python_rand_1 = random.random()
    torch_rand_1 = torch.rand(1).item()

    set_seed(123)
    python_rand_2 = random.random()
    torch_rand_2 = torch.rand(1).item()

    assert python_rand_1 == python_rand_2
    assert torch_rand_1 == torch_rand_2

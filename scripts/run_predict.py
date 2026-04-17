import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train.dataset import MinecraftBuildBagDataset, build_build_samples
from src.train.modeling import make_model, resolve_device
from src.visualize.prediction_visualizer import PredictionVisualizer

DEFAULT_TEST_JSON = ROOT / "data/processed/minecraft_builds_filtered_test.json"
DEFAULT_CHECKPOINT = ROOT / "outputs/best_multilabel_cnn.pt"
DEFAULT_OUTPUT = ROOT / "outputs/test_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-label prediction on Minecraft test set")
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--viz-output-dir",
        type=Path,
        default=ROOT / "outputs/prediction_visualization",
        help="Directory for prediction visualization outputs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--max-images-per-build",
        type=int,
        default=None,
        help="Override max images per build used by MIL model (default: checkpoint value).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0 ...",
    )
    return parser.parse_args()


def compute_micro_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required_keys = [
        "idx_to_tag",
        "tag_to_idx",
        "args",
        "model_arch",
        "backbone_name",
        "attention_mode",
        "class_thresholds",
        "max_images_per_build",
        "model_state_dict",
    ]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise KeyError(
            "Checkpoint missing required keys for classwise-only pipeline: "
            f"{missing_keys}"
        )

    idx_to_tag = checkpoint["idx_to_tag"]
    tag_to_idx = checkpoint["tag_to_idx"]
    ckpt_args = checkpoint["args"]

    required_arg_keys = ["dropout"]
    missing_arg_keys = [k for k in required_arg_keys if k not in ckpt_args]
    if missing_arg_keys:
        raise KeyError(
            "Checkpoint args missing required keys for classwise-only pipeline: "
            f"{missing_arg_keys}"
        )

    model_dropout = float(ckpt_args["dropout"])
    model_arch = str(checkpoint["model_arch"])
    backbone_name = str(checkpoint["backbone_name"])
    attention_mode = str(checkpoint["attention_mode"])

    if attention_mode != "class_specific_only":
        raise ValueError(f"Unsupported attention mode in checkpoint: {attention_mode}")

    ckpt_class_thresholds = checkpoint["class_thresholds"]
    if not isinstance(ckpt_class_thresholds, dict) or not ckpt_class_thresholds:
        raise ValueError("Checkpoint must contain non-empty class_thresholds.")

    threshold_values: list[float] = []
    for tag in idx_to_tag:
        if tag not in ckpt_class_thresholds:
            raise KeyError(f"Checkpoint class_thresholds missing tag: {tag}")
        threshold_values.append(float(ckpt_class_thresholds[tag]))

    ckpt_max_images = int(checkpoint["max_images_per_build"])
    max_images_per_build = args.max_images_per_build or ckpt_max_images

    samples = build_build_samples(args.test_json, tag_to_idx, ROOT)
    if not samples:
        raise ValueError("No valid test build samples found.")

    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_ds = MinecraftBuildBagDataset(
        samples,
        transform=eval_tf,
        max_images_per_build=max_images_per_build,
        train_mode=False,
        return_index=True,
    )

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = use_cuda

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    model = make_model(
        num_classes=len(idx_to_tag),
        dropout=model_dropout,
        backbone_name=backbone_name,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold_tensor = torch.tensor(threshold_values, device=device).unsqueeze(0)

    all_probs: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []

    with torch.no_grad():
        for images, mask, targets, indices in loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, mask)
                probs = torch.sigmoid(logits)

            preds = (probs >= threshold_tensor).float()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_indices.append(indices.cpu())

    probs_t = torch.cat(all_probs, dim=0)
    preds_t = torch.cat(all_preds, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    indices_t = torch.cat(all_indices, dim=0).tolist()

    metrics = compute_micro_metrics(preds_t, targets_t)

    prediction_rows: List[Dict] = []
    for out_i, sample_idx in enumerate(indices_t):
        sample = samples[sample_idx]
        image_paths = [str(p) for p in sample["image_paths"]]
        prob_vec = probs_t[out_i].tolist()
        pred_vec = preds_t[out_i].tolist()
        pred_tags = [idx_to_tag[j] for j, flag in enumerate(pred_vec) if flag >= 0.5]
        prediction_rows.append(
            {
                "title": sample["title"],
                "build_url": sample["build_url"],
                "image_path": image_paths[0] if image_paths else "",
                "image_paths": image_paths,
                "true_tags": sample["true_tags"],
                "predicted_tags": pred_tags,
                "probs": {idx_to_tag[j]: float(prob_vec[j]) for j in range(len(idx_to_tag))},
            }
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "test_json": str(args.test_json),
        "max_images_per_build": max_images_per_build,
        "attention_mode": attention_mode,
        "threshold_mode": "classwise",
        "applied_thresholds": {idx_to_tag[i]: float(threshold_values[i]) for i in range(len(idx_to_tag))},
        "num_test_builds": len(samples),
        "num_classes": len(idx_to_tag),
        "classes": idx_to_tag,
        "metrics": metrics,
        "predictions": prediction_rows,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    viz = PredictionVisualizer(result)
    viz_outputs = viz.save_visualizations(output_dir=str(args.viz_output_dir))

    min_th = min(threshold_values)
    max_th = max(threshold_values)
    avg_th = sum(threshold_values) / max(len(threshold_values), 1)

    print("=" * 80)
    print("Inference complete: Multi-label Minecraft Test Set (Build-level MIL)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Model arch: {model_arch}")
    print(f"Backbone: {backbone_name}")
    print(f"Model dropout: {model_dropout}")
    print(f"Attention mode: {attention_mode}")
    print("Threshold mode: classwise")
    print(f"Classwise thresholds: min={min_th:.2f}, mean={avg_th:.2f}, max={max_th:.2f}")
    print(f"Max images per build: {max_images_per_build}")
    print(f"Test builds: {len(samples)}")
    print(
        f"Metrics (classwise) | precision_micro={metrics['precision_micro']:.4f} | "
        f"recall_micro={metrics['recall_micro']:.4f} | "
        f"f1_micro={metrics['f1_micro']:.4f}"
    )
    print(f"Saved predictions: {args.output_file}")
    print(f"Saved prediction visualizations: {args.viz_output_dir}")
    for key, path in viz_outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()

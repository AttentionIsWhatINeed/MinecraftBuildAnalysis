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
        help="Override max images per build used by MIL model (default: checkpoint value or 6).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="auto",
        choices=["auto", "global", "classwise"],
        help="Threshold mode: auto(use checkpoint preference), global(single threshold), classwise(per-tag thresholds).",
    )
    parser.add_argument(
        "--use-checkpoint-threshold",
        action="store_true",
        default=True,
        help="Use checkpoint objective/global threshold for inference.",
    )
    parser.add_argument(
        "--no-use-checkpoint-threshold",
        action="store_false",
        dest="use_checkpoint_threshold",
        help="Ignore checkpoint threshold and force --threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.set_defaults(amp=True)
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
    idx_to_tag = checkpoint["idx_to_tag"]
    tag_to_idx = checkpoint["tag_to_idx"]
    ckpt_args = checkpoint.get("args", {})
    model_dropout = float(ckpt_args.get("dropout", checkpoint.get("dropout", 0.2)))
    model_arch = str(ckpt_args.get("model_arch", checkpoint.get("model_arch", "pretrained_mil")))
    backbone_name = str(ckpt_args.get("backbone_name", checkpoint.get("backbone_name", "resnet18")))
    class_specific_attention = bool(
        ckpt_args.get("class_specific_attention", checkpoint.get("class_specific_attention", False))
    )
    ckpt_global_threshold = float(
        checkpoint.get("objective_threshold", checkpoint.get("global_threshold", args.threshold))
    )
    ckpt_threshold_mode = str(checkpoint.get("threshold_mode", "global")).lower()
    ckpt_class_thresholds = checkpoint.get("class_thresholds", None)
    has_ckpt_class_thresholds = isinstance(ckpt_class_thresholds, dict) and len(ckpt_class_thresholds) > 0

    ckpt_max_images = checkpoint.get("max_images_per_build")
    if ckpt_max_images is None:
        ckpt_max_images = checkpoint.get("args", {}).get("max_images_per_build", 6)
    max_images_per_build = args.max_images_per_build or int(ckpt_max_images)

    if args.use_checkpoint_threshold:
        if args.threshold_mode == "global":
            selected_mode = "global"
        elif args.threshold_mode == "classwise":
            if not has_ckpt_class_thresholds:
                raise ValueError("Checkpoint has no class_thresholds, cannot use --threshold-mode classwise.")
            selected_mode = "classwise"
        else:
            if has_ckpt_class_thresholds and ckpt_threshold_mode == "classwise":
                selected_mode = "classwise"
            else:
                selected_mode = "global"

        if selected_mode == "classwise":
            threshold_values = [float(ckpt_class_thresholds.get(tag, ckpt_global_threshold)) for tag in idx_to_tag]
            effective_global_threshold = None
            threshold_source = "checkpoint_classwise"
        else:
            effective_global_threshold = float(ckpt_global_threshold)
            threshold_values = [effective_global_threshold] * len(idx_to_tag)
            threshold_source = "checkpoint_global"
    else:
        if args.threshold_mode == "classwise":
            raise ValueError("--threshold-mode classwise requires checkpoint thresholds; remove --no-use-checkpoint-threshold.")

        selected_mode = "global"
        effective_global_threshold = float(args.threshold)
        threshold_values = [effective_global_threshold] * len(idx_to_tag)
        threshold_source = "cli_global"

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
    use_amp = bool(args.amp and use_cuda)

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
        pretrained_backbone=False,
        class_specific_attention=class_specific_attention,
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
    if selected_mode == "global":
        metrics_mode_label = f"global={effective_global_threshold:.2f}"
    else:
        metrics_mode_label = "classwise"

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
        "threshold": args.threshold,
        "class_specific_attention": class_specific_attention,
        "threshold_mode_arg": args.threshold_mode,
        "checkpoint_threshold_mode": ckpt_threshold_mode,
        "checkpoint_has_class_thresholds": has_ckpt_class_thresholds,
        "effective_global_threshold": effective_global_threshold,
        "checkpoint_global_threshold": ckpt_global_threshold,
        "threshold_mode": selected_mode,
        "metrics_mode": selected_mode,
        "threshold_source": threshold_source,
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

    print("=" * 80)
    print("Inference complete: Multi-label Minecraft Test Set (Build-level MIL)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Model arch: {model_arch}")
    if model_arch == "pretrained_mil":
        print(f"Backbone: {backbone_name}")
    print(f"Model dropout: {model_dropout}")
    print(f"Class-specific attention: {class_specific_attention}")
    print(f"Threshold mode: {selected_mode}")
    if selected_mode == "global":
        print(f"Global threshold: {effective_global_threshold}")
    else:
        min_th = min(threshold_values)
        max_th = max(threshold_values)
        avg_th = sum(threshold_values) / max(len(threshold_values), 1)
        print(f"Classwise thresholds: min={min_th:.2f}, mean={avg_th:.2f}, max={max_th:.2f}")
    print(f"Max images per build: {max_images_per_build}")
    print(f"Test builds: {len(samples)}")
    print(
        f"Metrics ({metrics_mode_label}) | precision_micro={metrics['precision_micro']:.4f} | "
        f"recall_micro={metrics['recall_micro']:.4f} | "
        f"f1_micro={metrics['f1_micro']:.4f}"
    )
    print(f"Saved predictions: {args.output_file}")
    print(f"Saved prediction visualizations: {args.viz_output_dir}")
    for key, path in viz_outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()

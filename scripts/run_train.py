import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train.dataset import MinecraftBuildBagDataset, build_datasets
from src.train.engine import run_epoch
from src.train.modeling import make_model, resolve_device
from src.train.utils import set_seed


DEFAULT_TRAIN_JSON = ROOT / "data/processed/minecraft_builds_filtered_train.json"
DEFAULT_VAL_JSON = ROOT / "data/processed/minecraft_builds_filtered_val.json"
DEFAULT_TEST_JSON = ROOT / "data/processed/minecraft_builds_filtered_test.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
THRESHOLD_GRID_MIN = 0.10
THRESHOLD_GRID_MAX = 0.90
THRESHOLD_GRID_STEP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-label MIL model for Minecraft build tags")
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="Backbone for pretrained MIL model.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        default=True,
        help="Load ImageNet pretrained weights for pretrained backbone.",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_false",
        dest="pretrained_backbone",
        help="Disable pretrained ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Freeze pretrained backbone for first N epochs before full finetuning.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Optional LR for backbone parameters (default: same as --lr)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hard-sample-report-file",
        type=Path,
        default=None,
        help="Optional JSON path to save top high-loss samples for data quality inspection.",
    )
    parser.add_argument(
        "--hard-sample-topk",
        type=int,
        default=30,
        help="Number of high-loss samples to keep in hard-sample report.",
    )
    parser.add_argument(
        "--hard-sample-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to analyze when exporting hard-sample report.",
    )
    parser.add_argument(
        "--max-images-per-build",
        type=int,
        default=6,
        help="Max images per build in each training bag.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=160,
        help="Save a training snapshot every N optimizer steps (0 disables snapshots)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory for step snapshots (default: <output-dir>/snapshots)",
    )
    parser.add_argument(
        "--loss-plot-file",
        type=Path,
        default=None,
        help="Path to latest loss curve image (default: <output-dir>/latest_loss_curve.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.set_defaults(amp=True)
    return parser.parse_args()


class TagSetMatchLoss(nn.Module):
    """Differentiable set-match objective: maximize sample-wise F1 over tags."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        sample_f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        return 1.0 - sample_f1.mean()


def _save_step_snapshot(
    snapshot_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    tag_to_idx: dict,
    idx_to_tag: list,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tag_to_idx": tag_to_idx,
            "idx_to_tag": idx_to_tag,
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
        },
        snapshot_path,
    )


def _plot_latest_loss_curve(
    loss_plot_file: Path,
    step_losses: list[float],
    train_epoch_losses: list[float],
    val_epoch_losses: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    loss_plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    if step_losses:
        step_axis = list(range(1, len(step_losses) + 1))
        axes[0].plot(step_axis, step_losses, color="#3E7CB1", linewidth=1.2)
    axes[0].set_title("Training Step Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    if train_epoch_losses:
        epoch_axis = list(range(1, len(train_epoch_losses) + 1))
        axes[1].plot(epoch_axis, train_epoch_losses, marker="o", label="train_loss", color="#2A9D8F")
    if val_epoch_losses:
        epoch_axis = list(range(1, len(val_epoch_losses) + 1))
        axes[1].plot(epoch_axis, val_epoch_losses, marker="o", label="val_loss", color="#E76F51")
    axes[1].set_title("Epoch Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    if train_epoch_losses or val_epoch_losses:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(loss_plot_file, dpi=200)
    plt.close(fig)


def _collect_probs_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, mask, targets in loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                probs = torch.sigmoid(model(images, mask))

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    if not all_probs:
        raise ValueError("No batches found when collecting probabilities.")

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def _compute_micro_metrics_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
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


def _compute_set_match_metrics_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - preds) * targets).sum(dim=1)

    sample_f1 = (2.0 * tp + 1e-9) / (2.0 * tp + fp + fn + 1e-9)
    exact_match = (preds == targets).all(dim=1).float()

    return {
        "sample_f1": float(sample_f1.mean().item()),
        "exact_match_ratio": float(exact_match.mean().item()),
    }


def _per_sample_set_match_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - probs) * targets).sum(dim=1)
    sample_f1 = (2.0 * tp + 1e-9) / (2.0 * tp + fp + fn + 1e-9)
    return 1.0 - sample_f1


def _make_threshold_candidates() -> list[float]:
    count = int(round((THRESHOLD_GRID_MAX - THRESHOLD_GRID_MIN) / THRESHOLD_GRID_STEP)) + 1
    return [round(THRESHOLD_GRID_MIN + i * THRESHOLD_GRID_STEP, 4) for i in range(count)]


def _search_global_threshold_for_set_match(
    probs: torch.Tensor,
    targets: torch.Tensor,
    default_threshold: float,
) -> tuple[float, float]:
    candidates = _make_threshold_candidates()
    best_th = float(default_threshold)
    best_score = -1.0

    for th in candidates:
        preds = (probs >= th).float()
        score = _compute_set_match_metrics_from_preds(preds, targets)["sample_f1"]

        if score > best_score + 1e-12:
            best_score = score
            best_th = th
        elif abs(score - best_score) <= 1e-12 and abs(th - default_threshold) < abs(best_th - default_threshold):
            best_th = th

    return float(best_th), float(best_score)


def _export_hard_sample_report(
    model: nn.Module,
    loader: DataLoader,
    samples: list[dict],
    idx_to_tag: list[str],
    device: torch.device,
    use_amp: bool,
    output_file: Path,
    top_k: int,
) -> None:
    model.eval()
    rows: list[dict] = []

    with torch.no_grad():
        for images, mask, targets, indices in loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, mask)
                probs = torch.sigmoid(logits)

            per_sample_losses = _per_sample_set_match_loss_from_probs(probs, targets)

            probs_cpu = probs.cpu()
            losses_cpu = per_sample_losses.cpu()
            indices_cpu = indices.cpu().tolist()

            for local_i, sample_idx in enumerate(indices_cpu):
                sample = samples[sample_idx]
                prob_vec = probs_cpu[local_i].tolist()
                top3 = sorted(
                    ((idx_to_tag[i], float(prob_vec[i])) for i in range(len(idx_to_tag))),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]

                rows.append(
                    {
                        "loss": float(losses_cpu[local_i].item()),
                        "title": sample.get("title", ""),
                        "build_url": sample.get("build_url", ""),
                        "num_images": len(sample.get("image_paths", [])),
                        "true_tags": sample.get("true_tags", []),
                        "top3_pred_probs": [{"tag": tag, "prob": prob} for tag, prob in top3],
                    }
                )

    rows.sort(key=lambda x: x["loss"], reverse=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_rows": len(rows),
                "top_k": top_k,
                "samples": rows[: max(top_k, 1)],
            },
            f,
            indent=2,
        )


def _build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder.backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.snapshot_dir or (args.output_dir / "snapshots")
    loss_plot_file = args.loss_plot_file or (args.output_dir / "latest_loss_curve.png")

    bundle = build_datasets(args.train_json, args.val_json, args.test_json, ROOT)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = MinecraftBuildBagDataset(
        bundle.train_build_samples,
        transform=train_tf,
        max_images_per_build=args.max_images_per_build,
        train_mode=True,
    )
    val_ds = MinecraftBuildBagDataset(
        bundle.val_build_samples,
        transform=eval_tf,
        max_images_per_build=args.max_images_per_build,
        train_mode=False,
    )
    test_ds = MinecraftBuildBagDataset(
        bundle.test_build_samples,
        transform=eval_tf,
        max_images_per_build=args.max_images_per_build,
        train_mode=False,
    )

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(args.amp and use_cuda)
    torch.backends.cudnn.benchmark = use_cuda

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    hard_sample_loader = None
    hard_sample_source = None
    if args.hard_sample_report_file is not None:
        if args.hard_sample_split == "train":
            hard_sample_source = bundle.train_build_samples
        else:
            hard_sample_source = bundle.val_build_samples

        hard_sample_ds = MinecraftBuildBagDataset(
            hard_sample_source,
            transform=eval_tf,
            max_images_per_build=args.max_images_per_build,
            train_mode=False,
            return_index=True,
        )
        hard_sample_loader = DataLoader(
            hard_sample_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
        )

    model = make_model(
        num_classes=len(bundle.idx_to_tag),
        dropout=args.dropout,
        backbone_name=args.backbone_name,
        pretrained_backbone=bool(args.pretrained_backbone),
    ).to(device)

    backbone_frozen = False
    if args.freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)
        backbone_frozen = True

    scaler = GradScaler(device.type, enabled=use_amp)

    criterion = TagSetMatchLoss()
    optimizer = _build_optimizer(model, args)

    optimizer_name = "adamw"
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr
    best_val_loss = float("inf")
    best_val_objective = -1.0
    epochs_without_improve = 0
    best_ckpt = args.output_dir / "best_multilabel_cnn.pt"

    print("=" * 80)
    print("Neural Network Training: Multi-label Minecraft Tag Classifier")
    print("=" * 80)
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP enabled: {use_amp}")
    print("Model arch: pretrained_mil")
    print(f"Backbone: {args.backbone_name}")
    print(f"Pretrained backbone: {bool(args.pretrained_backbone)}")
    print(f"Freeze backbone epochs: {args.freeze_backbone_epochs}")
    print(f"Head LR: {args.lr}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Grad accumulation steps: {args.grad_accum_steps}")
    print(f"Max grad norm: {args.max_grad_norm}")
    if args.hard_sample_report_file is not None:
        print(
            f"Hard-sample report: split={args.hard_sample_split}, top_k={args.hard_sample_topk}, "
            f"path={args.hard_sample_report_file}"
        )
    print(f"Max images per build: {args.max_images_per_build}")
    print(f"Train builds: {len(train_ds)}, Val builds: {len(val_ds)}, Test builds: {len(test_ds)}")
    print(f"Classes ({len(bundle.idx_to_tag)}): {bundle.idx_to_tag}")
    if args.snapshot_interval > 0:
        print(f"Snapshot interval: every {args.snapshot_interval} steps")
        print(f"Snapshot dir: {snapshot_dir}")
    print(f"Latest loss plot: {loss_plot_file}")

    history = []
    step_losses: list[float] = []
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if backbone_frozen and epoch > args.freeze_backbone_epochs:
            model.set_backbone_trainable(True)
            backbone_frozen = False
            print(f"[Unfreeze] Backbone is now trainable at epoch {epoch}.")

        model.train()
        train_running_loss = 0.0
        train_running_count = 0
        optimizer.zero_grad(set_to_none=True)
        num_train_batches = max(len(train_loader), 1)

        for batch_idx, (images, mask, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, mask)
                loss = criterion(logits, targets)

            loss_to_backprop = loss / max(args.grad_accum_steps, 1)
            scaler.scale(loss_to_backprop).backward()

            batch_size = images.size(0)
            loss_value = loss.item()
            train_running_loss += loss_value * batch_size
            train_running_count += batch_size
            step_losses.append(loss_value)

            should_step = (batch_idx % max(args.grad_accum_steps, 1) == 0) or (batch_idx == num_train_batches)
            if should_step:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            if should_step and args.snapshot_interval > 0 and global_step % args.snapshot_interval == 0:
                snapshot_path = snapshot_dir / f"snapshot_step_{global_step:07d}.pt"
                _save_step_snapshot(
                    snapshot_path=snapshot_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    tag_to_idx=bundle.tag_to_idx,
                    idx_to_tag=bundle.idx_to_tag,
                    args=args,
                    epoch=epoch,
                    global_step=global_step,
                )
                _plot_latest_loss_curve(
                    loss_plot_file=loss_plot_file,
                    step_losses=step_losses,
                    train_epoch_losses=train_epoch_losses,
                    val_epoch_losses=val_epoch_losses,
                )
                print(f"[Snapshot] Saved at step {global_step}: {snapshot_path}")

        train_loss = train_running_loss / max(train_running_count, 1)
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_probs_epoch, val_targets_epoch = _collect_probs_targets(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
        )
        val_preds_global_epoch = (val_probs_epoch >= args.threshold).float()
        val_metrics_global_epoch = _compute_micro_metrics_from_preds(val_preds_global_epoch, val_targets_epoch)
        val_set_global_epoch = _compute_set_match_metrics_from_preds(val_preds_global_epoch, val_targets_epoch)

        epoch_objective_threshold, _ = _search_global_threshold_for_set_match(
            probs=val_probs_epoch,
            targets=val_targets_epoch,
            default_threshold=args.threshold,
        )
        val_preds_objective_epoch = (val_probs_epoch >= epoch_objective_threshold).float()
        val_metrics_objective_epoch = _compute_micro_metrics_from_preds(
            val_preds_objective_epoch,
            val_targets_epoch,
        )
        val_set_objective_epoch = _compute_set_match_metrics_from_preds(
            val_preds_objective_epoch,
            val_targets_epoch,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "objective_threshold_epoch": epoch_objective_threshold,
            "precision_micro_global": val_metrics_global_epoch["precision_micro"],
            "recall_micro_global": val_metrics_global_epoch["recall_micro"],
            "f1_micro_global": val_metrics_global_epoch["f1_micro"],
            "sample_f1_global": val_set_global_epoch["sample_f1"],
            "exact_match_global": val_set_global_epoch["exact_match_ratio"],
            "precision_micro_objective": val_metrics_objective_epoch["precision_micro"],
            "recall_micro_objective": val_metrics_objective_epoch["recall_micro"],
            "f1_micro_objective": val_metrics_objective_epoch["f1_micro"],
            "sample_f1_objective": val_set_objective_epoch["sample_f1"],
            "exact_match_objective": val_set_objective_epoch["exact_match_ratio"],
        }
        history.append(row)
        train_epoch_losses.append(train_loss)
        val_epoch_losses.append(val_loss)

        _plot_latest_loss_curve(
            loss_plot_file=loss_plot_file,
            step_losses=step_losses,
            train_epoch_losses=train_epoch_losses,
            val_epoch_losses=val_epoch_losses,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"step={global_step} | "
            f"lr={current_lr:.6g} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_sample_f1_objective={val_set_objective_epoch['sample_f1']:.4f} | "
            f"val_sample_f1_global={val_set_global_epoch['sample_f1']:.4f}"
        )

        objective_score = val_set_objective_epoch["sample_f1"]
        if objective_score > best_val_objective + 1e-12:
            best_val_objective = objective_score
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tag_to_idx": bundle.tag_to_idx,
                    "idx_to_tag": bundle.idx_to_tag,
                    "args": vars(args),
                    "grad_accum_steps": args.grad_accum_steps,
                    "max_grad_norm": args.max_grad_norm,
                    "dropout": args.dropout,
                    "model_arch": "pretrained_mil",
                    "backbone_name": args.backbone_name,
                    "pretrained_backbone": bool(args.pretrained_backbone),
                    "freeze_backbone_epochs": args.freeze_backbone_epochs,
                    "max_images_per_build": args.max_images_per_build,
                    "best_val_loss": best_val_loss,
                    "best_val_objective": best_val_objective,
                    "objective_threshold": float(epoch_objective_threshold),
                },
                best_ckpt,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_probs, val_targets = _collect_probs_targets(model, val_loader, device=device, use_amp=use_amp)
    test_probs, test_targets = _collect_probs_targets(model, test_loader, device=device, use_amp=use_amp)

    objective_threshold, _ = _search_global_threshold_for_set_match(
        probs=val_probs,
        targets=val_targets,
        default_threshold=float(checkpoint.get("objective_threshold", args.threshold)),
    )

    val_preds_global = (val_probs >= args.threshold).float()
    val_preds_objective = (val_probs >= objective_threshold).float()

    test_preds_global = (test_probs >= args.threshold).float()
    test_preds_objective = (test_probs >= objective_threshold).float()

    val_metrics_global = _compute_micro_metrics_from_preds(val_preds_global, val_targets)
    val_metrics_objective = _compute_micro_metrics_from_preds(val_preds_objective, val_targets)
    test_metrics_global = _compute_micro_metrics_from_preds(test_preds_global, test_targets)
    test_metrics = _compute_micro_metrics_from_preds(test_preds_objective, test_targets)

    val_set_global = _compute_set_match_metrics_from_preds(val_preds_global, val_targets)
    val_set_objective = _compute_set_match_metrics_from_preds(val_preds_objective, val_targets)
    test_set_global = _compute_set_match_metrics_from_preds(test_preds_global, test_targets)
    test_set_objective = _compute_set_match_metrics_from_preds(test_preds_objective, test_targets)

    checkpoint["global_threshold"] = float(objective_threshold)
    checkpoint["objective_threshold"] = float(objective_threshold)
    checkpoint["primary_metric_mode"] = "global_set_match"
    checkpoint["threshold_grid"] = {
        "min": THRESHOLD_GRID_MIN,
        "max": THRESHOLD_GRID_MAX,
        "step": THRESHOLD_GRID_STEP,
    }
    checkpoint["val_metrics_global_threshold"] = val_metrics_global
    checkpoint["val_set_match_global_threshold"] = val_set_global
    checkpoint["val_metrics_objective_threshold"] = val_metrics_objective
    checkpoint["val_set_match_objective_threshold"] = val_set_objective
    torch.save(checkpoint, best_ckpt)

    print(
        "Validation metrics (objective threshold, preferred) | "
        f"precision_micro={val_metrics_objective['precision_micro']:.4f} | "
        f"recall_micro={val_metrics_objective['recall_micro']:.4f} | "
        f"f1_micro={val_metrics_objective['f1_micro']:.4f} | "
        f"sample_f1={val_set_objective['sample_f1']:.4f} | "
        f"exact_match={val_set_objective['exact_match_ratio']:.4f} | "
        f"threshold={objective_threshold:.2f}"
    )
    print(
        f"Validation metrics (global={args.threshold:.2f}, reference) | "
        f"precision_micro={val_metrics_global['precision_micro']:.4f} | "
        f"recall_micro={val_metrics_global['recall_micro']:.4f} | "
        f"f1_micro={val_metrics_global['f1_micro']:.4f} | "
        f"sample_f1={val_set_global['sample_f1']:.4f} | "
        f"exact_match={val_set_global['exact_match_ratio']:.4f}"
    )
    print(
        "Test metrics (objective threshold, preferred) | "
        f"precision_micro={test_metrics['precision_micro']:.4f} | "
        f"recall_micro={test_metrics['recall_micro']:.4f} | "
        f"f1_micro={test_metrics['f1_micro']:.4f} | "
        f"sample_f1={test_set_objective['sample_f1']:.4f} | "
        f"exact_match={test_set_objective['exact_match_ratio']:.4f}"
    )
    print(
        f"Test metrics (global={args.threshold:.2f}, reference) | "
        f"precision_micro={test_metrics_global['precision_micro']:.4f} | "
        f"recall_micro={test_metrics_global['recall_micro']:.4f} | "
        f"f1_micro={test_metrics_global['f1_micro']:.4f} | "
        f"sample_f1={test_set_global['sample_f1']:.4f} | "
        f"exact_match={test_set_global['exact_match_ratio']:.4f}"
    )

    if args.hard_sample_report_file is not None and hard_sample_loader is not None and hard_sample_source is not None:
        _export_hard_sample_report(
            model=model,
            loader=hard_sample_loader,
            samples=hard_sample_source,
            idx_to_tag=bundle.idx_to_tag,
            device=device,
            use_amp=use_amp,
            output_file=args.hard_sample_report_file,
            top_k=args.hard_sample_topk,
        )
        print(f"Saved hard-sample report: {args.hard_sample_report_file}")

    metrics_out = {
        "model_arch": "pretrained_mil",
        "backbone_name": args.backbone_name,
        "pretrained_backbone": bool(args.pretrained_backbone),
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "max_images_per_build": args.max_images_per_build,
        "dropout": args.dropout,
        "optimizer": optimizer_name,
        "lr": args.lr,
        "backbone_lr": backbone_lr,
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "hard_sample_report_file": str(args.hard_sample_report_file) if args.hard_sample_report_file else None,
        "hard_sample_topk": args.hard_sample_topk,
        "hard_sample_split": args.hard_sample_split,
        "global_threshold": float(args.threshold),
        "objective_threshold": float(objective_threshold),
        "primary_metric_mode": "global_set_match",
        "threshold_grid": {
            "min": THRESHOLD_GRID_MIN,
            "max": THRESHOLD_GRID_MAX,
            "step": THRESHOLD_GRID_STEP,
        },
        "best_val_loss": best_val_loss,
        "best_val_objective": best_val_objective,
        "val_metrics_global_threshold": val_metrics_global,
        "val_set_match_global_threshold": val_set_global,
        "val_metrics_objective_threshold": val_metrics_objective,
        "val_set_match_objective_threshold": val_set_objective,
        "test_metrics_global_threshold": test_metrics_global,
        "test_set_match_global_threshold": test_set_global,
        "test_metrics": test_metrics,
        "test_set_match_metrics": test_set_objective,
        "history": history,
        "train_step_losses": step_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
        "classes": bundle.idx_to_tag,
        "checkpoint": str(best_ckpt),
        "latest_loss_plot": str(loss_plot_file),
    }

    metrics_file = args.output_dir / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Saved checkpoint: {best_ckpt}")
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    main()

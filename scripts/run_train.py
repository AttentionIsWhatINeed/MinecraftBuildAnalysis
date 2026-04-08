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
from src.train.engine import evaluate_f1_micro, run_epoch
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
    parser = argparse.ArgumentParser(description="Train multi-label CNN for Minecraft build tags")
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau", "cosine"],
        help="Learning-rate scheduler strategy.",
    )
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-cosine-t-max", type=int, default=20)
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


def _save_step_snapshot(
    snapshot_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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


def _make_threshold_candidates() -> list[float]:
    count = int(round((THRESHOLD_GRID_MAX - THRESHOLD_GRID_MIN) / THRESHOLD_GRID_STEP)) + 1
    return [round(THRESHOLD_GRID_MIN + i * THRESHOLD_GRID_STEP, 4) for i in range(count)]


def _search_class_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
    idx_to_tag: list[str],
    default_threshold: float,
) -> Dict[str, float]:
    candidates = _make_threshold_candidates()
    thresholds: Dict[str, float] = {}

    for class_idx, tag in enumerate(idx_to_tag):
        class_probs = probs[:, class_idx]
        class_targets = targets[:, class_idx]

        best_th = float(default_threshold)
        best_f1 = -1.0

        for th in candidates:
            class_preds = (class_probs >= th).float()
            tp = (class_preds * class_targets).sum().item()
            fp = (class_preds * (1.0 - class_targets)).sum().item()
            fn = ((1.0 - class_preds) * class_targets).sum().item()

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_th = th
            elif abs(f1 - best_f1) <= 1e-12 and abs(th - default_threshold) < abs(best_th - default_threshold):
                best_th = th

        thresholds[tag] = float(best_th)

    return thresholds


def _predict_with_class_thresholds(
    probs: torch.Tensor,
    idx_to_tag: list[str],
    class_thresholds: Dict[str, float],
    default_threshold: float,
) -> torch.Tensor:
    threshold_values = [float(class_thresholds.get(tag, default_threshold)) for tag in idx_to_tag]
    threshold_tensor = torch.tensor(threshold_values, dtype=probs.dtype).unsqueeze(0)
    return (probs >= threshold_tensor).float()


def _build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.lr_scheduler == "none":
        return None

    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.lr_cosine_t_max, 1),
        eta_min=args.lr_scheduler_min_lr,
    )


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
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    rows: list[dict] = []

    with torch.no_grad():
        for images, mask, targets, indices in loader:
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, mask)
                probs = torch.sigmoid(logits)

            per_sample_losses = criterion(logits, targets).mean(dim=1)

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

    model = make_model(num_classes=len(bundle.idx_to_tag), dropout=args.dropout).to(device)
    scaler = GradScaler(device.type, enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, args)

    best_val_loss = float("inf")
    epochs_without_improve = 0
    best_ckpt = args.output_dir / "best_multilabel_cnn.pt"

    print("=" * 80)
    print("Neural Network Training: Multi-label Minecraft Tag Classifier")
    print("=" * 80)
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP enabled: {use_amp}")
    print(f"Dropout: {args.dropout}")
    print(f"Grad accumulation steps: {args.grad_accum_steps}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"LR scheduler: {args.lr_scheduler}")
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
                    scheduler=scheduler,
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
        val_metrics = evaluate_f1_micro(
            model,
            val_loader,
            device,
            threshold=args.threshold,
            use_amp=use_amp,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
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
            f"val_f1_micro={val_metrics['f1_micro']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tag_to_idx": bundle.tag_to_idx,
                    "idx_to_tag": bundle.idx_to_tag,
                    "args": vars(args),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "grad_accum_steps": args.grad_accum_steps,
                    "max_grad_norm": args.max_grad_norm,
                    "dropout": args.dropout,
                    "max_images_per_build": args.max_images_per_build,
                    "best_val_loss": best_val_loss,
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

    class_thresholds = _search_class_thresholds(
        probs=val_probs,
        targets=val_targets,
        idx_to_tag=bundle.idx_to_tag,
        default_threshold=args.threshold,
    )

    val_preds_global = (val_probs >= args.threshold).float()
    val_preds_classwise = _predict_with_class_thresholds(
        probs=val_probs,
        idx_to_tag=bundle.idx_to_tag,
        class_thresholds=class_thresholds,
        default_threshold=args.threshold,
    )

    test_preds_global = (test_probs >= args.threshold).float()
    test_preds_classwise = _predict_with_class_thresholds(
        probs=test_probs,
        idx_to_tag=bundle.idx_to_tag,
        class_thresholds=class_thresholds,
        default_threshold=args.threshold,
    )

    val_metrics_global = _compute_micro_metrics_from_preds(val_preds_global, val_targets)
    val_metrics_classwise = _compute_micro_metrics_from_preds(val_preds_classwise, val_targets)
    test_metrics_global = _compute_micro_metrics_from_preds(test_preds_global, test_targets)
    test_metrics = _compute_micro_metrics_from_preds(test_preds_classwise, test_targets)

    checkpoint["class_thresholds"] = class_thresholds
    checkpoint["global_threshold"] = float(args.threshold)
    checkpoint["threshold_grid"] = {
        "min": THRESHOLD_GRID_MIN,
        "max": THRESHOLD_GRID_MAX,
        "step": THRESHOLD_GRID_STEP,
    }
    checkpoint["val_metrics_global_threshold"] = val_metrics_global
    checkpoint["val_metrics_classwise_threshold"] = val_metrics_classwise
    torch.save(checkpoint, best_ckpt)

    print(
        f"Validation metrics (global={args.threshold:.2f}) | "
        f"precision_micro={val_metrics_global['precision_micro']:.4f} | "
        f"recall_micro={val_metrics_global['recall_micro']:.4f} | "
        f"f1_micro={val_metrics_global['f1_micro']:.4f}"
    )
    print(
        "Validation metrics (classwise thresholds) | "
        f"precision_micro={val_metrics_classwise['precision_micro']:.4f} | "
        f"recall_micro={val_metrics_classwise['recall_micro']:.4f} | "
        f"f1_micro={val_metrics_classwise['f1_micro']:.4f}"
    )
    print(
        f"Test metrics (global={args.threshold:.2f}) | "
        f"precision_micro={test_metrics_global['precision_micro']:.4f} | "
        f"recall_micro={test_metrics_global['recall_micro']:.4f} | "
        f"f1_micro={test_metrics_global['f1_micro']:.4f}"
    )
    print(
        "Test metrics (classwise thresholds) | "
        f"precision_micro={test_metrics['precision_micro']:.4f} | "
        f"recall_micro={test_metrics['recall_micro']:.4f} | "
        f"f1_micro={test_metrics['f1_micro']:.4f}"
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
        "max_images_per_build": args.max_images_per_build,
        "dropout": args.dropout,
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "lr_cosine_t_max": args.lr_cosine_t_max,
        "hard_sample_report_file": str(args.hard_sample_report_file) if args.hard_sample_report_file else None,
        "hard_sample_topk": args.hard_sample_topk,
        "hard_sample_split": args.hard_sample_split,
        "global_threshold": float(args.threshold),
        "class_thresholds": class_thresholds,
        "threshold_grid": {
            "min": THRESHOLD_GRID_MIN,
            "max": THRESHOLD_GRID_MAX,
            "step": THRESHOLD_GRID_STEP,
        },
        "best_val_loss": best_val_loss,
        "val_metrics_global_threshold": val_metrics_global,
        "val_metrics_classwise_threshold": val_metrics_classwise,
        "test_metrics_global_threshold": test_metrics_global,
        "test_metrics": test_metrics,
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

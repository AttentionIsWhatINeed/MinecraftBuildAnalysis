import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train.dataset import MinecraftImageDataset, build_datasets
from src.train.engine import evaluate_f1_micro, run_epoch
from src.train.modeling import make_model, resolve_device
from src.train.utils import set_seed


DEFAULT_TRAIN_JSON = ROOT / "data/processed/minecraft_builds_filtered_train.json"
DEFAULT_VAL_JSON = ROOT / "data/processed/minecraft_builds_filtered_val.json"
DEFAULT_TEST_JSON = ROOT / "data/processed/minecraft_builds_filtered_test.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-label CNN for Minecraft build tags")
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--val-json", type=Path, default=DEFAULT_VAL_JSON)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=50,
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

    train_ds = MinecraftImageDataset(bundle.train_samples, transform=train_tf)
    val_ds = MinecraftImageDataset(bundle.val_samples, transform=eval_tf)
    test_ds = MinecraftImageDataset(bundle.test_samples, transform=eval_tf)

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

    model = make_model(num_classes=len(bundle.idx_to_tag)).to(device)
    scaler = GradScaler(device.type, enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
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
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            loss_value = loss.item()
            train_running_loss += loss_value * batch_size
            train_running_count += batch_size

            global_step += 1
            step_losses.append(loss_value)

            if args.snapshot_interval > 0 and global_step % args.snapshot_interval == 0:
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
        val_metrics = evaluate_f1_micro(
            model,
            val_loader,
            device,
            threshold=args.threshold,
            use_amp=use_amp,
        )

        row = {
            "epoch": epoch,
            "global_step": global_step,
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

    test_metrics = evaluate_f1_micro(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        use_amp=use_amp,
    )
    print(
        f"Test metrics | precision_micro={test_metrics['precision_micro']:.4f} | "
        f"recall_micro={test_metrics['recall_micro']:.4f} | "
        f"f1_micro={test_metrics['f1_micro']:.4f}"
    )

    metrics_out = {
        "best_val_loss": best_val_loss,
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

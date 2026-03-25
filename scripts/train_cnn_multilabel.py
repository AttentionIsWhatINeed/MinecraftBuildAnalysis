import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler
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
        "--device",
        type=str,
        default="auto",
        help="Training device: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
        )
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
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
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
        "classes": bundle.idx_to_tag,
        "checkpoint": str(best_ckpt),
    }

    metrics_file = args.output_dir / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Saved checkpoint: {best_ckpt}")
    print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    main()

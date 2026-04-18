import argparse
import json
import logging
import sys
from datetime import datetime
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
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Freeze backbone for first N epochs before full finetuning.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Optional LR for backbone parameters (default: 0.1 * --lr).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--asl-gamma-neg",
        type=float,
        default=4.0,
        help="ASL negative focusing parameter.",
    )
    parser.add_argument(
        "--asl-gamma-pos",
        type=float,
        default=0.0,
        help="ASL positive focusing parameter.",
    )
    parser.add_argument(
        "--asl-clip",
        type=float,
        default=0.05,
        help="ASL probability clip applied to negative probabilities.",
    )
    parser.add_argument(
        "--asl-eps",
        type=float,
        default=1e-8,
        help="ASL epsilon for numerical stability.",
    )
    parser.add_argument("--seed", type=int, default=13)
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
        "--log-file",
        type=Path,
        default=None,
        help="File path for training debug log (default: <output-dir>/train_debug.log)",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="JSONL file for structured training events (default: <output-dir>/training_events.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from state file.",
    )
    parser.add_argument(
        "--resume-state-file-path",
        type=Path,
        default=None,
        help="Path to saved training state file (default: <output-dir>/last_training_state.pt)",
    )
    parser.add_argument(
        "--save-state-file-path",
        type=Path,
        default=None,
        help="Where to save resumable training state (default: <output-dir>/last_training_state.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto | cpu | cuda | cuda:0 ...",
    )
    return parser.parse_args()


class AsymmetricLossMultiLabel(nn.Module):
    """ASL for multi-label classification."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos

        if self.clip > 0:
            probs_neg = torch.clamp(probs_neg + self.clip, max=1.0)

        probs_pos = torch.clamp(probs_pos, min=self.eps, max=1.0)
        probs_neg = torch.clamp(probs_neg, min=self.eps, max=1.0)

        loss = targets * torch.log(probs_pos) + (1.0 - targets) * torch.log(probs_neg)

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)

        return -loss.mean()


def _setup_file_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def _log(logger: logging.Logger, msg: str) -> None:
    print(msg)
    logger.info(msg)


def _append_event(events_file: Path, payload: dict) -> None:
    events_file.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(), **payload}
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_training_state(
    state_file: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    bundle,
    epoch: int,
    epoch_completed: bool,
    global_step: int,
    best_val_loss: float,
    best_val_objective: float,
    epochs_without_improve: int,
    history: list[dict],
    step_losses: list[float],
    train_epoch_losses: list[float],
    val_epoch_losses: list[float],
    backbone_frozen: bool,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_version": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tag_to_idx": bundle.tag_to_idx,
            "idx_to_tag": bundle.idx_to_tag,
            "args": vars(args),
            "epoch": epoch,
            "epoch_completed": epoch_completed,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_val_objective": best_val_objective,
            "epochs_without_improve": epochs_without_improve,
            "history": history,
            "train_step_losses": step_losses,
            "train_epoch_losses": train_epoch_losses,
            "val_epoch_losses": val_epoch_losses,
            "backbone_frozen": backbone_frozen,
        },
        state_file,
    )


def _serialize_args(args: argparse.Namespace) -> dict:
    serialized = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            serialized[k] = str(v)
        else:
            serialized[k] = v
    return serialized


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


def _compute_macro_metrics_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision_macro": float(precision.mean().item()),
        "recall_macro": float(recall.mean().item()),
        "f1_macro": float(f1.mean().item()),
    }


def _compute_binary_f1_for_class(pred_col: torch.Tensor, target_col: torch.Tensor) -> float:
    tp = (pred_col * target_col).sum().item()
    fp = (pred_col * (1.0 - target_col)).sum().item()
    fn = ((1.0 - pred_col) * target_col).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return float(f1)


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


def _make_threshold_candidates() -> list[float]:
    count = int(round((THRESHOLD_GRID_MAX - THRESHOLD_GRID_MIN) / THRESHOLD_GRID_STEP)) + 1
    return [round(THRESHOLD_GRID_MIN + i * THRESHOLD_GRID_STEP, 4) for i in range(count)]


def _apply_threshold_vector(probs: torch.Tensor, threshold_values: list[float]) -> torch.Tensor:
    threshold_tensor = torch.tensor(threshold_values, dtype=probs.dtype, device=probs.device).unsqueeze(0)
    return (probs >= threshold_tensor).float()


def _search_classwise_thresholds_for_binary_f1(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[float], float]:
    candidates = _make_threshold_candidates()
    num_classes = int(probs.size(1))
    thresholds = [float(candidates[0])] * num_classes

    per_class_f1_scores: list[float] = []

    for class_idx in range(num_classes):
        target_col = targets[:, class_idx]
        local_best_th = float(candidates[0])
        local_best_score = -1.0

        for th in candidates:
            pred_col = (probs[:, class_idx] >= th).float()
            trial_score = _compute_binary_f1_for_class(pred_col, target_col)

            if trial_score > local_best_score + 1e-12:
                local_best_score = trial_score
                local_best_th = th

        thresholds[class_idx] = float(local_best_th)
        per_class_f1_scores.append(float(local_best_score))

    macro_f1 = sum(per_class_f1_scores) / max(len(per_class_f1_scores), 1)
    return thresholds, float(macro_f1)


def _resolve_backbone_lr(args: argparse.Namespace) -> float:
    if args.backbone_lr is not None:
        return float(args.backbone_lr)

    return float(args.lr) * 0.1


def _build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_lr = _resolve_backbone_lr(args)
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
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

    if args.asl_gamma_neg < 0 or args.asl_gamma_pos < 0:
        raise ValueError("--asl-gamma-neg and --asl-gamma-pos must be >= 0")
    if args.asl_eps <= 0:
        raise ValueError("--asl-eps must be > 0")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.snapshot_dir or (args.output_dir / "snapshots")
    loss_plot_file = args.loss_plot_file or (args.output_dir / "latest_loss_curve.png")
    log_file = args.log_file or (args.output_dir / "train_debug.log")
    events_file = args.events_file or (args.output_dir / "training_events.jsonl")
    save_state_file_path = args.save_state_file_path or (args.output_dir / "last_training_state.pt")
    resume_state_file_path = args.resume_state_file_path or save_state_file_path

    if args.resume and not resume_state_file_path.exists():
        raise FileNotFoundError(f"Resume requested, but state file not found: {resume_state_file_path}")

    logger = _setup_file_logger(log_file)
    _append_event(
        events_file,
        {
            "event": "run_start",
            "args": _serialize_args(args),
            "log_file": str(log_file),
            "events_file": str(events_file),
            "save_state_file_path": str(save_state_file_path),
            "resume_state_file_path": str(resume_state_file_path),
        },
    )

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
    use_amp = use_cuda
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

    model = make_model(
        num_classes=len(bundle.idx_to_tag),
        dropout=args.dropout,
        backbone_name=args.backbone_name,
    ).to(device)

    backbone_frozen = False
    if args.freeze_backbone_epochs > 0:
        model.set_backbone_trainable(False)
        backbone_frozen = True

    scaler = GradScaler(device.type, enabled=use_amp)
    criterion = AsymmetricLossMultiLabel(
        gamma_neg=args.asl_gamma_neg,
        gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
        eps=args.asl_eps,
    )
    optimizer = _build_optimizer(model, args)

    optimizer_name = "adamw"
    backbone_lr = _resolve_backbone_lr(args)
    best_val_loss = float("inf")
    best_val_objective = -1.0
    epochs_without_improve = 0
    best_ckpt = args.output_dir / "best_multilabel_cnn.pt"

    history: list[dict] = []
    step_losses: list[float] = []
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []
    global_step = 0
    start_epoch = 1

    if args.resume:
        resume_state = torch.load(resume_state_file_path, map_location=device, weights_only=False)

        resume_idx_to_tag = resume_state["idx_to_tag"]
        if list(resume_idx_to_tag) != list(bundle.idx_to_tag):
            raise ValueError(
                "Resume state classes do not match current dataset classes. "
                f"State classes={resume_idx_to_tag}, current classes={bundle.idx_to_tag}"
            )

        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])

        history = list(resume_state["history"])
        step_losses = list(resume_state["train_step_losses"])
        train_epoch_losses = list(resume_state["train_epoch_losses"])
        val_epoch_losses = list(resume_state["val_epoch_losses"])

        global_step = int(resume_state["global_step"])
        best_val_loss = float(resume_state["best_val_loss"])
        best_val_objective = float(resume_state["best_val_objective"])
        epochs_without_improve = int(resume_state["epochs_without_improve"])

        resume_epoch = int(resume_state["epoch"])
        epoch_completed = bool(resume_state["epoch_completed"])
        start_epoch = max(resume_epoch + 1, 1) if epoch_completed else max(resume_epoch, 1)

        backbone_frozen = bool(resume_state["backbone_frozen"])
        model.set_backbone_trainable(not backbone_frozen)

        _log(logger, f"[Resume] Loaded state: {resume_state_file_path}")
        _log(
            logger,
            f"[Resume] start_epoch={start_epoch}, global_step={global_step}, "
            f"best_val_objective={best_val_objective:.4f}",
        )
        _append_event(
            events_file,
            {
                "event": "resume_loaded",
                "resume_state_file_path": str(resume_state_file_path),
                "start_epoch": start_epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_objective": best_val_objective,
                "history_rows": len(history),
            },
        )

    _log(logger, "=" * 80)
    _log(logger, "Neural Network Training: Multi-label Minecraft Tag Classifier")
    _log(logger, "=" * 80)
    _log(logger, f"Device: {device}")
    if use_cuda:
        _log(logger, f"GPU: {torch.cuda.get_device_name(device)}")
    _log(logger, f"AMP enabled: {use_amp}")
    _log(logger, "Model arch: pretrained_mil")
    _log(logger, f"Backbone: {args.backbone_name}")
    _log(logger, "Attention mode: class_specific_only")
    _log(logger, f"Freeze backbone epochs: {args.freeze_backbone_epochs}")
    _log(logger, f"Head LR: {args.lr}")
    _log(logger, f"Backbone LR: {backbone_lr}")
    _log(logger, f"Weight decay: {args.weight_decay}")
    _log(logger, f"Dropout: {args.dropout}")
    _log(logger, f"Optimizer: {optimizer_name}")
    _log(logger, f"Grad accumulation steps: {args.grad_accum_steps}")
    _log(logger, f"Max grad norm: {args.max_grad_norm}")
    _log(
        logger,
        f"Loss: ASL(gamma_neg={args.asl_gamma_neg:.3f}, gamma_pos={args.asl_gamma_pos:.3f}, "
        f"clip={args.asl_clip:.3f}, eps={args.asl_eps:.1e})",
    )
    _log(logger, f"Max images per build: {args.max_images_per_build}")
    _log(logger, f"Train builds: {len(train_ds)}, Val builds: {len(val_ds)}, Test builds: {len(test_ds)}")
    _log(logger, f"Classes ({len(bundle.idx_to_tag)}): {bundle.idx_to_tag}")
    if args.snapshot_interval > 0:
        _log(logger, f"Snapshot interval: every {args.snapshot_interval} steps")
        _log(logger, f"Snapshot dir: {snapshot_dir}")
    _log(logger, f"Latest loss plot: {loss_plot_file}")
    _log(logger, f"Debug log file: {log_file}")
    _log(logger, f"Events file: {events_file}")
    _log(logger, f"State file: {save_state_file_path}")

    current_epoch = max(start_epoch - 1, 0)
    interrupted = False

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            if backbone_frozen and epoch > args.freeze_backbone_epochs:
                model.set_backbone_trainable(True)
                backbone_frozen = False
                _log(logger, f"[Unfreeze] Backbone is now trainable at epoch {epoch}.")

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

                loss_to_backprop = loss / args.grad_accum_steps
                scaler.scale(loss_to_backprop).backward()

                batch_size = images.size(0)
                loss_value = loss.item()
                train_running_loss += loss_value * batch_size
                train_running_count += batch_size
                step_losses.append(loss_value)

                should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == num_train_batches)
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
                    _save_training_state(
                        state_file=save_state_file_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        bundle=bundle,
                        epoch=epoch,
                        epoch_completed=False,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        best_val_objective=best_val_objective,
                        epochs_without_improve=epochs_without_improve,
                        history=history,
                        step_losses=step_losses,
                        train_epoch_losses=train_epoch_losses,
                        val_epoch_losses=val_epoch_losses,
                        backbone_frozen=backbone_frozen,
                    )
                    _plot_latest_loss_curve(
                        loss_plot_file=loss_plot_file,
                        step_losses=step_losses,
                        train_epoch_losses=train_epoch_losses,
                        val_epoch_losses=val_epoch_losses,
                    )
                    _log(logger, f"[Snapshot] Saved at step {global_step}: {snapshot_path}")
                    _append_event(
                        events_file,
                        {
                            "event": "snapshot_saved",
                            "epoch": epoch,
                            "global_step": global_step,
                            "snapshot_path": str(snapshot_path),
                        },
                    )

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
            class_threshold_values_epoch, _ = _search_classwise_thresholds_for_binary_f1(
                probs=val_probs_epoch,
                targets=val_targets_epoch,
            )
            val_preds_classwise_epoch = _apply_threshold_vector(val_probs_epoch, class_threshold_values_epoch)
            val_metrics_classwise_epoch = _compute_micro_metrics_from_preds(val_preds_classwise_epoch, val_targets_epoch)
            val_macro_classwise_epoch = _compute_macro_metrics_from_preds(val_preds_classwise_epoch, val_targets_epoch)
            val_set_classwise_epoch = _compute_set_match_metrics_from_preds(val_preds_classwise_epoch, val_targets_epoch)

            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "class_threshold_min": min(class_threshold_values_epoch),
                "class_threshold_mean": sum(class_threshold_values_epoch) / max(len(class_threshold_values_epoch), 1),
                "class_threshold_max": max(class_threshold_values_epoch),
                "precision_micro_class_threshold": val_metrics_classwise_epoch["precision_micro"],
                "recall_micro_class_threshold": val_metrics_classwise_epoch["recall_micro"],
                "f1_micro_class_threshold": val_metrics_classwise_epoch["f1_micro"],
                "precision_macro_class_threshold": val_macro_classwise_epoch["precision_macro"],
                "recall_macro_class_threshold": val_macro_classwise_epoch["recall_macro"],
                "f1_macro_class_threshold": val_macro_classwise_epoch["f1_macro"],
                "sample_f1_class_threshold": val_set_classwise_epoch["sample_f1"],
                "exact_match_class_threshold": val_set_classwise_epoch["exact_match_ratio"],
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

            _log(
                logger,
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"step={global_step} | "
                f"lr={current_lr:.6g} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_sample_f1_class_threshold={val_set_classwise_epoch['sample_f1']:.4f}",
            )
            _append_event(events_file, {"event": "epoch_end", **row})

            objective_score = val_set_classwise_epoch["sample_f1"]
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
                        "attention_mode": "class_specific_only",
                        "freeze_backbone_epochs": args.freeze_backbone_epochs,
                        "max_images_per_build": args.max_images_per_build,
                        "best_val_loss": best_val_loss,
                        "best_val_objective": best_val_objective,
                        "class_thresholds": {
                            bundle.idx_to_tag[i]: float(class_threshold_values_epoch[i])
                            for i in range(len(bundle.idx_to_tag))
                        },
                        "threshold_grid": {
                            "min": THRESHOLD_GRID_MIN,
                            "max": THRESHOLD_GRID_MAX,
                            "step": THRESHOLD_GRID_STEP,
                        },
                    },
                    best_ckpt,
                )
                _append_event(
                    events_file,
                    {
                        "event": "best_checkpoint_updated",
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_objective": best_val_objective,
                        "best_ckpt": str(best_ckpt),
                    },
                )
            else:
                epochs_without_improve += 1

            _save_training_state(
                state_file=save_state_file_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                bundle=bundle,
                epoch=epoch,
                epoch_completed=True,
                global_step=global_step,
                best_val_loss=best_val_loss,
                best_val_objective=best_val_objective,
                epochs_without_improve=epochs_without_improve,
                history=history,
                step_losses=step_losses,
                train_epoch_losses=train_epoch_losses,
                val_epoch_losses=val_epoch_losses,
                backbone_frozen=backbone_frozen,
            )

            if epochs_without_improve >= args.patience:
                _log(logger, f"Early stopping triggered after {epoch} epochs.")
                _append_event(events_file, {"event": "early_stop", "epoch": epoch})
                break
    except KeyboardInterrupt:
        interrupted = True
        _log(logger, "[Interrupt] KeyboardInterrupt captured, saving resumable state...")
        _save_training_state(
            state_file=save_state_file_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            bundle=bundle,
            epoch=max(current_epoch, 0),
            epoch_completed=False,
            global_step=global_step,
            best_val_loss=best_val_loss,
            best_val_objective=best_val_objective,
            epochs_without_improve=epochs_without_improve,
            history=history,
            step_losses=step_losses,
            train_epoch_losses=train_epoch_losses,
            val_epoch_losses=val_epoch_losses,
            backbone_frozen=backbone_frozen,
        )
        _append_event(
            events_file,
            {
                "event": "interrupt_saved",
                "epoch": max(current_epoch, 0),
                "global_step": global_step,
                "save_state_file_path": str(save_state_file_path),
            },
        )

    if interrupted:
        _log(logger, f"Saved resumable state: {save_state_file_path}")
        _log(logger, "Training interrupted. Resume with: python scripts/run_train.py --resume")
        return

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found after training: {best_ckpt}")

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_probs, val_targets = _collect_probs_targets(model, val_loader, device=device, use_amp=use_amp)
    test_probs, test_targets = _collect_probs_targets(model, test_loader, device=device, use_amp=use_amp)

    class_threshold_values, _ = _search_classwise_thresholds_for_binary_f1(
        probs=val_probs,
        targets=val_targets,
    )

    val_preds_classwise = _apply_threshold_vector(val_probs, class_threshold_values)
    test_preds_classwise = _apply_threshold_vector(test_probs, class_threshold_values)

    val_metrics_classwise = _compute_micro_metrics_from_preds(val_preds_classwise, val_targets)
    test_metrics_classwise = _compute_micro_metrics_from_preds(test_preds_classwise, test_targets)

    val_macro_classwise = _compute_macro_metrics_from_preds(val_preds_classwise, val_targets)
    test_macro_classwise = _compute_macro_metrics_from_preds(test_preds_classwise, test_targets)

    val_set_classwise = _compute_set_match_metrics_from_preds(val_preds_classwise, val_targets)
    test_set_classwise = _compute_set_match_metrics_from_preds(test_preds_classwise, test_targets)

    checkpoint["attention_mode"] = "class_specific_only"
    checkpoint["class_thresholds"] = {
        bundle.idx_to_tag[i]: float(class_threshold_values[i]) for i in range(len(bundle.idx_to_tag))
    }
    checkpoint["threshold_grid"] = {
        "min": THRESHOLD_GRID_MIN,
        "max": THRESHOLD_GRID_MAX,
        "step": THRESHOLD_GRID_STEP,
    }
    checkpoint["val_metrics_class_threshold"] = val_metrics_classwise
    checkpoint["val_macro_metrics_class_threshold"] = val_macro_classwise
    checkpoint["val_set_match_class_threshold"] = val_set_classwise
    checkpoint["test_metrics_class_threshold"] = test_metrics_classwise
    checkpoint["test_macro_metrics_class_threshold"] = test_macro_classwise
    checkpoint["test_set_match_class_threshold"] = test_set_classwise
    torch.save(checkpoint, best_ckpt)

    _log(
        logger,
        "Validation metrics (classwise thresholds, preferred) | "
        f"precision_micro={val_metrics_classwise['precision_micro']:.4f} | "
        f"recall_micro={val_metrics_classwise['recall_micro']:.4f} | "
        f"f1_micro={val_metrics_classwise['f1_micro']:.4f} | "
        f"precision_macro={val_macro_classwise['precision_macro']:.4f} | "
        f"recall_macro={val_macro_classwise['recall_macro']:.4f} | "
        f"f1_macro={val_macro_classwise['f1_macro']:.4f} | "
        f"sample_f1={val_set_classwise['sample_f1']:.4f} | "
        f"exact_match={val_set_classwise['exact_match_ratio']:.4f}"
    )
    _log(
        logger,
        "Test metrics (classwise thresholds, preferred) | "
        f"precision_micro={test_metrics_classwise['precision_micro']:.4f} | "
        f"recall_micro={test_metrics_classwise['recall_micro']:.4f} | "
        f"f1_micro={test_metrics_classwise['f1_micro']:.4f} | "
        f"precision_macro={test_macro_classwise['precision_macro']:.4f} | "
        f"recall_macro={test_macro_classwise['recall_macro']:.4f} | "
        f"f1_macro={test_macro_classwise['f1_macro']:.4f} | "
        f"sample_f1={test_set_classwise['sample_f1']:.4f} | "
        f"exact_match={test_set_classwise['exact_match_ratio']:.4f}"
    )

    metrics_out = {
        "model_arch": "pretrained_mil",
        "backbone_name": args.backbone_name,
        "attention_mode": "class_specific_only",
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "resume": bool(args.resume),
        "resumed_from": str(resume_state_file_path) if args.resume else None,
        "max_images_per_build": args.max_images_per_build,
        "dropout": args.dropout,
        "optimizer": optimizer_name,
        "loss_name": "asl",
        "asl_gamma_neg": args.asl_gamma_neg,
        "asl_gamma_pos": args.asl_gamma_pos,
        "asl_clip": args.asl_clip,
        "asl_eps": args.asl_eps,
        "lr": args.lr,
        "backbone_lr": backbone_lr,
        "weight_decay": args.weight_decay,
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "class_thresholds": {
            bundle.idx_to_tag[i]: float(class_threshold_values[i]) for i in range(len(bundle.idx_to_tag))
        },
        "threshold_grid": {
            "min": THRESHOLD_GRID_MIN,
            "max": THRESHOLD_GRID_MAX,
            "step": THRESHOLD_GRID_STEP,
        },
        "best_val_loss": best_val_loss,
        "best_val_objective": best_val_objective,
        "val_metrics_class_threshold": val_metrics_classwise,
        "val_macro_metrics_class_threshold": val_macro_classwise,
        "val_set_match_class_threshold": val_set_classwise,
        "test_metrics": test_metrics_classwise,
        "test_macro_metrics": test_macro_classwise,
        "test_set_match_metrics": test_set_classwise,
        "test_metrics_class_threshold": test_metrics_classwise,
        "test_macro_metrics_class_threshold": test_macro_classwise,
        "test_set_match_class_threshold": test_set_classwise,
        "history": history,
        "train_step_losses": step_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
        "classes": bundle.idx_to_tag,
        "checkpoint": str(best_ckpt),
        "save_state_file_path": str(save_state_file_path),
        "debug_log_file": str(log_file),
        "events_file": str(events_file),
        "latest_loss_plot": str(loss_plot_file),
    }

    metrics_file = args.output_dir / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    final_epoch = int(history[-1]["epoch"]) if history else max(current_epoch, 0)
    _save_training_state(
        state_file=save_state_file_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        bundle=bundle,
        epoch=final_epoch,
        epoch_completed=True,
        global_step=global_step,
        best_val_loss=best_val_loss,
        best_val_objective=best_val_objective,
        epochs_without_improve=epochs_without_improve,
        history=history,
        step_losses=step_losses,
        train_epoch_losses=train_epoch_losses,
        val_epoch_losses=val_epoch_losses,
        backbone_frozen=backbone_frozen,
    )

    _append_event(
        events_file,
        {
            "event": "run_end",
            "final_epoch": final_epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_val_objective": best_val_objective,
            "metrics_file": str(metrics_file),
            "best_ckpt": str(best_ckpt),
            "save_state_file_path": str(save_state_file_path),
        },
    )

    _log(logger, f"Saved checkpoint: {best_ckpt}")
    _log(logger, f"Saved metrics: {metrics_file}")
    _log(logger, f"Saved state: {save_state_file_path}")


if __name__ == "__main__":
    main()

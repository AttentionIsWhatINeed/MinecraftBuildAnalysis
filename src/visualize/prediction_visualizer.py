import json
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class PredictionVisualizer:
    """Create plots and summary files for multi-label prediction results."""

    DEFAULT_EXAMPLE_COUNT = 12

    def __init__(self, result: Dict):
        self.result = result
        self.classes: List[str] = result.get("classes", [])
        self.predictions: List[Dict] = result.get("predictions", [])

        if not self.classes:
            raise ValueError("Prediction result has no classes.")
        if not self.predictions:
            raise ValueError("Prediction result has no prediction rows.")

    def _compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}

        for tag in self.classes:
            tp = 0
            fp = 0
            fn = 0
            true_count = 0
            pred_count = 0

            for row in self.predictions:
                true_tags = set(row.get("true_tags", []))
                pred_tags = set(row.get("predicted_tags", []))
                in_true = tag in true_tags
                in_pred = tag in pred_tags

                if in_true:
                    true_count += 1
                if in_pred:
                    pred_count += 1

                if in_true and in_pred:
                    tp += 1
                elif (not in_true) and in_pred:
                    fp += 1
                elif in_true and (not in_pred):
                    fn += 1

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            metrics[tag] = {
                "true_count": float(true_count),
                "pred_count": float(pred_count),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

        return metrics

    def _compute_cardinality_stats(self) -> Dict[str, float]:
        true_cardinality = [len(row.get("true_tags", [])) for row in self.predictions]
        pred_cardinality = [len(row.get("predicted_tags", [])) for row in self.predictions]

        exact_match = 0
        for row in self.predictions:
            if set(row.get("true_tags", [])) == set(row.get("predicted_tags", [])):
                exact_match += 1

        n = max(len(self.predictions), 1)
        return {
            "avg_true_tags_per_image": sum(true_cardinality) / n,
            "avg_pred_tags_per_image": sum(pred_cardinality) / n,
            "exact_match_ratio": exact_match / n,
        }

    def _prepare_example_rows(self, max_examples: int) -> List[Dict]:
        max_examples = max(1, int(max_examples))
        mismatched: List[Dict] = []
        matched: List[Dict] = []

        for row in self.predictions:
            true_tags = set(row.get("true_tags", []))
            pred_tags = set(row.get("predicted_tags", []))
            if true_tags == pred_tags:
                matched.append(row)
            else:
                mismatched.append(row)

        display_rows = mismatched[:max_examples]
        if len(display_rows) < max_examples:
            display_rows.extend(matched[: max_examples - len(display_rows)])
        return display_rows

    @staticmethod
    def _get_display_image_path(row: Dict) -> str:
        image_path = str(row.get("image_path", "") or "").strip()
        if image_path:
            return image_path

        image_paths = row.get("image_paths", [])
        if isinstance(image_paths, list) and image_paths:
            return str(image_paths[0]).strip()
        return ""

    @staticmethod
    def _format_tags(tags: List[str], max_items: int = 8) -> str:
        safe_tags = [str(t) for t in tags]
        if not safe_tags:
            return "(none)"
        if len(safe_tags) <= max_items:
            return ", ".join(safe_tags)
        remain = len(safe_tags) - max_items
        return ", ".join(safe_tags[:max_items]) + f", ... (+{remain})"

    @staticmethod
    def _build_colored_tag_line_box(label: str, tags: List[str], matched_tags: set, max_items: int = 8):
        from matplotlib.offsetbox import HPacker, TextArea

        parts = [
            TextArea(
                f"{label}:",
                textprops={
                    "fontsize": 8,
                    "fontweight": "bold",
                    "color": "black",
                },
            )
        ]

        safe_tags = [str(t) for t in tags]
        if not safe_tags:
            parts.append(TextArea(" (none)", textprops={"fontsize": 8, "color": "#666666"}))
            return HPacker(children=parts, align="baseline", pad=0, sep=2)

        display_tags = safe_tags[:max_items]
        for idx, tag in enumerate(display_tags):
            color = "#2E7D32" if tag in matched_tags else "#C62828"
            prefix = " " if idx == 0 else ", "
            parts.append(TextArea(f"{prefix}{tag}", textprops={"fontsize": 8, "color": color}))

        remain = len(safe_tags) - len(display_tags)
        if remain > 0:
            parts.append(TextArea(f", ... (+{remain})", textprops={"fontsize": 8, "color": "#555555"}))

        return HPacker(children=parts, align="baseline", pad=0, sep=0)

    def save_visualizations(
        self,
        output_dir: str = "outputs/prediction_visualization",
        example_count: int = DEFAULT_EXAMPLE_COUNT,
    ) -> Dict[str, str]:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.offsetbox import AnchoredOffsetbox, VPacker
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            ) from exc

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {}
        per_class = self._compute_per_class_metrics()
        cardinality_stats = self._compute_cardinality_stats()

        # Plot 1: micro metrics bar chart
        micro = self.result.get("metrics", {})
        micro_names = ["precision_micro", "recall_micro", "f1_micro"]
        micro_vals = [float(micro.get(k, 0.0)) for k in micro_names]

        plt.figure(figsize=(7, 5))
        bars = plt.bar(micro_names, micro_vals, color=["#3E7CB1", "#2A9D8F", "#E76F51"])
        plt.ylim(0, 1)
        plt.title("Prediction Metrics (Micro)")
        plt.ylabel("Score")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, val in zip(bars, micro_vals):
            plt.text(bar.get_x() + bar.get_width() / 2, min(val + 0.02, 0.98), f"{val:.3f}", ha="center")
        path_micro = out / "prediction_micro_metrics.png"
        plt.tight_layout()
        plt.savefig(path_micro, dpi=200)
        plt.close()
        saved["micro_metrics"] = str(path_micro)

        # Plot 2: per-class support (true vs predicted)
        cls = self.classes
        true_counts = [per_class[t]["true_count"] for t in cls]
        pred_counts = [per_class[t]["pred_count"] for t in cls]

        x = list(range(len(cls)))
        width = 0.42
        plt.figure(figsize=(13, 6))
        plt.bar([v - width / 2 for v in x], true_counts, width=width, label="true_count", color="#2A9D8F")
        plt.bar([v + width / 2 for v in x], pred_counts, width=width, label="pred_count", color="#F4A261")
        plt.xticks(x, cls, rotation=35, ha="right")
        plt.ylabel("Count")
        plt.title("Per-class Support: True vs Predicted")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path_support = out / "prediction_per_class_support.png"
        plt.tight_layout()
        plt.savefig(path_support, dpi=200)
        plt.close()
        saved["per_class_support"] = str(path_support)

        # Plot 3: per-class F1
        f1_vals = [per_class[t]["f1"] for t in cls]
        plt.figure(figsize=(13, 6))
        plt.bar(cls, f1_vals, color="#E76F51")
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("F1")
        plt.title("Per-class F1 Score")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path_f1 = out / "prediction_per_class_f1.png"
        plt.tight_layout()
        plt.savefig(path_f1, dpi=200)
        plt.close()
        saved["per_class_f1"] = str(path_f1)

        # Plot 4: true/pred tag cardinality distribution
        true_card = [len(row.get("true_tags", [])) for row in self.predictions]
        pred_card = [len(row.get("predicted_tags", [])) for row in self.predictions]

        max_card = max(true_card + pred_card) if (true_card or pred_card) else 0
        bins = range(0, max_card + 2)

        plt.figure(figsize=(10, 6))
        plt.hist(true_card, bins=bins, alpha=0.6, label="true_tags_per_image", color="#3E7CB1", edgecolor="black")
        plt.hist(pred_card, bins=bins, alpha=0.6, label="pred_tags_per_image", color="#E9C46A", edgecolor="black")
        plt.xticks(range(0, max_card + 1))
        plt.xlabel("Tag Count Per Image")
        plt.ylabel("Number of Images")
        plt.title("Tag Cardinality Distribution")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path_card = out / "prediction_tag_cardinality_hist.png"
        plt.tight_layout()
        plt.savefig(path_card, dpi=200)
        plt.close()
        saved["tag_cardinality_hist"] = str(path_card)

        # Plot 5: limited examples with image + true/predicted tags
        example_rows = self._prepare_example_rows(example_count)
        cols = 3
        rows = max(1, int(math.ceil(len(example_rows) / cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.6))
        axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]

        for ax, row in zip(axes_list, example_rows):
            image_path = self._get_display_image_path(row)
            image_loaded = False

            if image_path:
                image_file = Path(image_path)
                if image_file.exists():
                    try:
                        ax.imshow(plt.imread(image_file))
                        image_loaded = True
                    except Exception:
                        image_loaded = False

            if not image_loaded:
                ax.set_facecolor("#F2F2F2")
                ax.text(0.5, 0.5, "Image not available", ha="center", va="center", fontsize=9)

            true_tags = row.get("true_tags", [])
            pred_tags = row.get("predicted_tags", [])

            title = row.get("title", "")
            ax.set_title(textwrap.fill(str(title), width=55), fontsize=8)

            true_set = set(true_tags)
            pred_set = set(pred_tags)
            true_line = self._build_colored_tag_line_box("True", true_tags, pred_set)
            pred_line = self._build_colored_tag_line_box("Pred", pred_tags, true_set)
            tags_box = VPacker(children=[true_line, pred_line], align="left", pad=0, sep=1)

            anchored = AnchoredOffsetbox(
                loc="lower left",
                child=tags_box,
                pad=0.2,
                frameon=True,
                bbox_to_anchor=(0.01, 0.01),
                bbox_transform=ax.transAxes,
                borderpad=0.2,
            )
            anchored.patch.set_facecolor("white")
            anchored.patch.set_alpha(0.85)
            anchored.patch.set_edgecolor("#DDDDDD")
            ax.add_artist(anchored)

            ax.axis("off")

        for ax in axes_list[len(example_rows) :]:
            ax.axis("off")

        plt.suptitle(f"Prediction Examples: Image with True vs Predicted Tags (Count: {len(example_rows)})")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path_examples = out / "prediction_examples_image_true_pred_tags.png"
        plt.savefig(path_examples, dpi=200)
        plt.close(fig)
        saved["image_true_pred_examples"] = str(path_examples)

        # Save summary JSON
        example_rows_summary = []
        for row in example_rows:
            true_tags = row.get("true_tags", [])
            pred_tags = row.get("predicted_tags", [])
            example_rows_summary.append(
                {
                    "title": row.get("title", ""),
                    "build_url": row.get("build_url", ""),
                    "image_path": self._get_display_image_path(row),
                    "true_tags": true_tags,
                    "predicted_tags": pred_tags,
                    "exact_match": bool(set(true_tags) == set(pred_tags)),
                }
            )

        summary_path = out / "prediction_viz_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "num_predictions": len(self.predictions),
                    "micro_metrics": {
                        "precision_micro": float(micro.get("precision_micro", 0.0)),
                        "recall_micro": float(micro.get("recall_micro", 0.0)),
                        "f1_micro": float(micro.get("f1_micro", 0.0)),
                    },
                    "cardinality": cardinality_stats,
                    "per_class_metrics": per_class,
                    "example_display": {
                        "example_count": len(example_rows),
                        "rows": example_rows_summary,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        saved["summary_json"] = str(summary_path)

        return saved

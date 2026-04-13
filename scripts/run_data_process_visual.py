"""
Generate before/after visualizations for data processing outputs.

This script is intentionally separate from run_data_process.py so that
visualization can be run independently after dataset generation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualize.data_process_visualizer import DataProcessVisualizer


DEFAULT_BEFORE_FILE = ROOT / "data/raw/metadata/builds_metadata.json"
DEFAULT_AFTER_FILE = ROOT / "data/processed/minecraft_builds_filtered.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run before/after data processing visualizations."
    )
    parser.add_argument(
        "--before-file",
        type=Path,
        default=DEFAULT_BEFORE_FILE,
        help="Path to raw metadata JSON used as 'before' data.",
    )
    parser.add_argument(
        "--after-file",
        type=Path,
        default=DEFAULT_AFTER_FILE,
        help="Path to processed JSON used as 'after' data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--before-top-n",
        type=int,
        default=20,
        help="Top N tags to show for before-processing ranking.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=8,
        help="Number of build examples to export for before/after.",
    )
    return parser.parse_args()


def _load_builds_from_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        builds = payload.get("builds")
        if isinstance(builds, list):
            return builds

    raise ValueError(f"Unsupported JSON structure for builds: {path}")


def main() -> None:
    args = parse_args()

    before_builds = _load_builds_from_json(args.before_file)
    after_builds = _load_builds_from_json(args.after_file)

    visualizer = DataProcessVisualizer()
    outputs = visualizer.save_before_after_visualizations(
        before_builds=before_builds,
        after_builds=after_builds,
        output_dir=str(args.output_dir),
        before_top_n=args.before_top_n,
        sample_n=args.sample_n,
    )

    print("=" * 80)
    print("DATA PROCESS VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Before file: {args.before_file}")
    print(f"After file: {args.after_file}")
    print(f"Output dir: {args.output_dir}")
    print("Generated files:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()

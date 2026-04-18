import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualize.raw_data_visualizer import RawDataVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw data visualization for Minecraft build metadata")
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="data/raw/metadata/builds_metadata.json",
        help="Path to raw metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/raw_visualization",
        help="Output directory for generated figures and summary",
    )
    parser.add_argument(
        "--top-n-tags",
        type=int,
        default=50,
        help="Number of top tags to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    visualizer = RawDataVisualizer(metadata_file=args.metadata_file)
    outputs = visualizer.save_visualizations(output_dir=args.output_dir, top_n_tags=args.top_n_tags)
    summary = visualizer.get_summary()

    print("=" * 80)
    print("RAW DATA VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Metadata file: {args.metadata_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total builds: {summary.total_builds}")
    print(f"Total images: {summary.total_images}")
    print(f"Unique tags: {summary.unique_tags}")
    print(f"Average images/build: {summary.avg_images_per_build:.2f}")
    print(f"Average tags/build: {summary.avg_tags_per_build:.2f}")
    print("Generated files:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()

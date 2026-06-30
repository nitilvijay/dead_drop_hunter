import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".pgm"}


def convert_single_image(source_path: Path, input_root: Path, output_root: Path) -> bool:
    relative_path = source_path.relative_to(input_root)
    destination_path = output_root / relative_path.with_suffix(".png")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(source_path) as image:
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")

            image.save(destination_path, format="PNG", optimize=True)
        return True
    except Exception as exc:
        print(f"Failed to convert {source_path}: {exc}")
        return False


def convert_tree_to_png(
    input_root: Path, output_root: Path, max_workers: int = 8
) -> tuple[int, int]:
    """Convert BMP and PGM images under input_root to PNG files under output_root."""

    source_files = [
        source_path
        for source_path in input_root.rglob("*")
        if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    skipped = sum(
        1
        for source_path in input_root.rglob("*")
        if source_path.is_file() and source_path.suffix.lower() not in IMAGE_EXTENSIONS
    )

    converted = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda source_path: convert_single_image(source_path, input_root, output_root),
            source_files,
        )
        for success in tqdm(
            results,
            total=len(source_files),
            desc=f"Converting {input_root.name}",
            unit="image",
        ):
            if success:
                converted += 1
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BMP and PGM images in train_tiles and test_tiles to PNG in a new folder"
    )
    parser.add_argument(
        "--base-dir",
        default="/home/nitil/brainfuel/dead_drop_hunter",
        help="Project root containing train_tiles and test_tiles with BMP/PGM images",
    )
    parser.add_argument(
        "--output-dir",
        default="compressed_tiles_png",
        help="Name of the new folder to store PNG copies",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads to use",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_root = base_dir / args.output_dir

    sources = ["train_tiles", "test_tiles"]
    total_converted = 0
    total_skipped = 0

    for folder_name in sources:
        input_root = base_dir / folder_name
        if not input_root.exists():
            print(f"Skipping missing folder: {input_root}")
            continue

        destination_root = output_root / folder_name
        converted, skipped = convert_tree_to_png(
            input_root, destination_root, max_workers=args.workers
        )
        total_converted += converted
        total_skipped += skipped
        print(f"{folder_name}: converted {converted} images")

    print(f"Done. PNG copies stored in: {output_root}")
    print(f"Total converted: {total_converted}")
    print(f"Total skipped non-image files: {total_skipped}")


if __name__ == "__main__":
    main()
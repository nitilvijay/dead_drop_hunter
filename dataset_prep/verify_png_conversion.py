"""
Verifies that PGM/BMP -> PNG conversion did not alter any pixel values.
Run this on a sample (or all) of your converted files before trusting the
PNG copy for training.

Usage:
    python verify_png_conversion.py /path/to/original_dir /path/to/png_dir
"""
import sys
import os
import glob
import numpy as np
from PIL import Image


def load_gray_array(path):
    with Image.open(path) as im:
        return np.array(im.convert("L"), dtype=np.int32)


def find_match(orig_path, png_dir):
    stem = os.path.splitext(os.path.basename(orig_path))[0]
    candidates = glob.glob(os.path.join(png_dir, stem + ".png"))
    return candidates[0] if candidates else None


def main(orig_dir, png_dir):
    originals = sorted(glob.glob(os.path.join(orig_dir, "**", "*.*"), recursive=True))
    originals = [p for p in originals if p.lower().endswith((".pgm", ".bmp"))]

    if not originals:
        print("No .pgm/.bmp files found under", orig_dir)
        return

    checked, mismatched, missing = 0, 0, 0

    for orig_path in originals:
        png_path = find_match(orig_path, png_dir)
        if png_path is None:
            missing += 1
            continue

        orig_arr = load_gray_array(orig_path)
        png_arr = load_gray_array(png_path)

        checked += 1
        if orig_arr.shape != png_arr.shape:
            mismatched += 1
            print(f"[SHAPE MISMATCH] {orig_path}: {orig_arr.shape} vs {png_arr.shape}")
            continue

        if not np.array_equal(orig_arr, png_arr):
            mismatched += 1
            diff = np.abs(orig_arr.astype(int) - png_arr.astype(int))
            print(f"[PIXEL MISMATCH] {orig_path}: max_diff={diff.max()}, "
                  f"changed_pixels={(diff > 0).sum()} / {diff.size}")

    print("\n--- Summary ---")
    print(f"Checked:   {checked}")
    print(f"Mismatched: {mismatched}")
    print(f"Missing PNG counterpart: {missing}")

    if mismatched == 0 and checked > 0:
        print("\nAll checked files are pixel-identical. Safe to use the PNG copy.")
    elif mismatched > 0:
        print("\nSome files differ at the pixel level. Do NOT use the PNG copy "
              "for those files without re-converting correctly.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_png_conversion.py <original_dir> <png_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
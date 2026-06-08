import os
import random
import shutil
import subprocess
from PIL import Image

def process_caltech_dataset():
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    source_dir = os.path.join(base_dir, "sp4g8h7v8k-1/CALTECH-BMP-1500")
    temp_jpeg_dir = os.path.join(base_dir, "CALTECH_grayscale_jpeg")
    train_jpeg_dir = os.path.join(base_dir, "train_jpeg")
    test_jpeg_dir = os.path.join(base_dir, "test_jpeg")

    # Create directories
    for d in [temp_jpeg_dir, train_jpeg_dir, test_jpeg_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # 1. Convert BMP to Grayscale and then to JPEG
    # We use a new range starting from 20001 to avoid collisions
    print("Converting BMP to Grayscale JPEG...")
    bmp_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".bmp")])
    full_jpegs = []
    
    for i, f in enumerate(bmp_files):
        new_name = f"{20001 + i}.jpg"
        src_path = os.path.join(source_dir, f)
        dest_path = os.path.join(temp_jpeg_dir, new_name)
        
        try:
            with Image.open(src_path) as img:
                gray_img = img.convert("L")
                # Initial JPEG compression - using quality 95 as a high-quality baseline
                gray_img.save(dest_path, "JPEG", quality=95)
            full_jpegs.append(new_name)
        except Exception as e:
            print(f"Error converting {f}: {e}")
            
        if (i + 1) % 500 == 0:
            print(f"Converted {i + 1}/{len(bmp_files)} images.")

    # 2. Random Split (80% Train, 20% Test)
    print("Performing 80/20 split on source images...")
    test_count = int(len(full_jpegs) * 0.2)
    test_sources = set(random.sample(full_jpegs, test_count))

    # 3. Lossless Tiling using jpegtran
    # This avoids recompression artifacts by cropping directly on the DCT coefficients
    print("Performing lossless tiling using jpegtran...")
    tile_size = 256
    
    for filename in full_jpegs:
        is_test = filename in test_sources
        target_dir = test_jpeg_dir if is_test else train_jpeg_dir
        src_path = os.path.join(temp_jpeg_dir, filename)
        base_name = os.path.splitext(filename)[0]

        # 512x512 -> 4 tiles of 256x256
        crops = [
            ("0", "0"),   # Tile 0
            ("256", "0"), # Tile 1
            ("0", "256"), # Tile 2
            ("256", "256")# Tile 3
        ]

        for idx, (x, y) in enumerate(crops):
            tile_filename = f"{base_name}_tile{idx}.jpg"
            dest_path = os.path.join(target_dir, tile_filename)
            
            # Use jpegtran for lossless cropping
            # -crop WxH+X+Y: crops to a specific region
            cmd = [
                "jpegtran",
                "-crop", f"{tile_size}x{tile_size}+{x}+{y}",
                "-outfile", dest_path,
                src_path
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error tiling {filename} at {x},{y}: {e}")

    print(f"Processing complete.")
    print(f"Train JPEG tiles: {len(os.listdir(train_jpeg_dir))}")
    print(f"Test JPEG tiles: {len(os.listdir(test_jpeg_dir))}")
    
    # Cleanup intermediate full-size JPEGs
    shutil.rmtree(temp_jpeg_dir)
    print(f"Cleaned up intermediate directory: {temp_jpeg_dir}")

if __name__ == "__main__":
    process_caltech_dataset()

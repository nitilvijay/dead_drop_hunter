import os
import random
import shutil
import subprocess
from PIL import Image

def process_clean_dataset():
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    source_dir = os.path.join(base_dir, "archive/val/val/clean")
    temp_jpeg_dir = os.path.join(base_dir, "CLEAN_grayscale_jpeg")
    train_jpeg_dir = os.path.join(base_dir, "train_jpeg")
    test_jpeg_dir = os.path.join(base_dir, "test_jpeg")

    # Ensure output directories exist
    for d in [train_jpeg_dir, test_jpeg_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    if not os.path.exists(temp_jpeg_dir):
        os.makedirs(temp_jpeg_dir)

    # 1. Convert PNG to Grayscale and then to JPEG
    # Use range 30001+ to avoid collisions
    print(f"Converting images from {source_dir} to Grayscale JPEG...")
    files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".bmp", ".jpg"))])
    full_jpegs = []
    
    for i, f in enumerate(files):
        new_name = f"{30001 + i}.jpg"
        src_path = os.path.join(source_dir, f)
        dest_path = os.path.join(temp_jpeg_dir, new_name)
        
        try:
            with Image.open(src_path) as img:
                gray_img = img.convert("L")
                gray_img.save(dest_path, "JPEG", quality=95)
            full_jpegs.append(new_name)
        except Exception as e:
            print(f"Error converting {f}: {e}")
            
        if (i + 1) % 500 == 0:
            print(f"Converted {i + 1}/{len(files)} images.")

    # 2. Random Split (80% Train, 20% Test)
    print("Performing 80/20 split on source images...")
    test_count = int(len(full_jpegs) * 0.2)
    test_sources = set(random.sample(full_jpegs, test_count))

    # 3. Lossless Tiling using jpegtran
    print("Performing lossless tiling using jpegtran...")
    tile_size = 256
    
    for filename in full_jpegs:
        is_test = filename in test_sources
        target_dir = test_jpeg_dir if is_test else train_jpeg_dir
        src_path = os.path.join(temp_jpeg_dir, filename)
        base_name = os.path.splitext(filename)[0]

        crops = [
            ("0", "0"),   
            ("256", "0"), 
            ("0", "256"), 
            ("256", "256")
        ]

        for idx, (x, y) in enumerate(crops):
            tile_filename = f"{base_name}_tile{idx}.jpg"
            dest_path = os.path.join(target_dir, tile_filename)
            
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
    # Show cumulative counts
    print(f"Cumulative Train JPEG tiles: {len(os.listdir(train_jpeg_dir))}")
    print(f"Cumulative Test JPEG tiles: {len(os.listdir(test_jpeg_dir))}")
    
    shutil.rmtree(temp_jpeg_dir)
    print(f"Cleaned up intermediate directory: {temp_jpeg_dir}")

if __name__ == "__main__":
    process_clean_dataset()

import os
import shutil
import random

def consolidate_and_split():
    # Paths
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    pgm_src = os.path.join(base_dir, "BOSSbase_1.01")
    bmp_src = os.path.join(base_dir, "sp4g8h7v8k-1/NRC-BMP-1500-grayscale")
    combined_dir = os.path.join(base_dir, "combined_grayscale_images")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Create directories
    for d in [combined_dir, train_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # 1. Consolidate PGM files (1-10000)
    print("Consolidating PGM files...")
    pgm_files = [f for f in os.listdir(pgm_src) if f.endswith(".pgm")]
    for f in pgm_files:
        shutil.copy2(os.path.join(pgm_src, f), os.path.join(combined_dir, f))

    # 2. Consolidate and Rename BMP files (10001-11500)
    print("Consolidating and renaming BMP files...")
    bmp_files = sorted([f for f in os.listdir(bmp_src) if f.endswith(".bmp")])
    for i, f in enumerate(bmp_files):
        new_name = f"{10001 + i}.bmp"
        shutil.copy2(os.path.join(bmp_src, f), os.path.join(combined_dir, new_name))

    # 3. Random Split (80% Train, 20% Test)
    # Total images = 10000 (PGM) + 1500 (BMP) = 11500
    all_combined_files = os.listdir(combined_dir)
    total_images = len(all_combined_files)
    test_count = int(total_images * 0.2)
    
    print(f"Total images: {total_images}. Target test set size: {test_count}")

    # Choose random files for test set
    test_files = random.sample(all_combined_files, test_count)
    test_files_set = set(test_files)

    # Move files to train and test folders
    print("Moving files to train and test folders...")
    for filename in all_combined_files:
        src_path = os.path.join(combined_dir, filename)
        if filename in test_files_set:
            shutil.move(src_path, os.path.join(test_dir, filename))
        else:
            shutil.move(src_path, os.path.join(train_dir, filename))

    print(f"Split complete. Train: {len(os.listdir(train_dir))}, Test: {len(os.listdir(test_dir))}")

if __name__ == "__main__":
    consolidate_and_split()

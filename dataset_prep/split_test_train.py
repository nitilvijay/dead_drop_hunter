import os
import shutil
import random
import re

def split():
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    source_dir = os.path.join(base_dir, "Combine_grayscale_images")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Create directories if they don't exist
    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # 1. Random Split (80% Train, 20% Test)
    all_files = os.listdir(source_dir)
    total_images = len(all_files)
    test_count = int(total_images * 0.2)
    
    print(f"Total images in {source_dir}: {total_images}")
    print(f"Target test set size: {test_count}")

    # Choose random files for test set
    test_files = random.sample(all_files, test_count)
    test_files_set = set(test_files)

    # Move files to train and test folders
    print("Moving files to train and test folders...")
    for filename in all_files:
        src_path = os.path.join(source_dir, filename)
        if filename in test_files_set:
            shutil.move(src_path, os.path.join(test_dir, filename))
        else:
            shutil.move(src_path, os.path.join(train_dir, filename))

    print(f"Split complete.")
    print(f"Train folder: {len(os.listdir(train_dir))} images")
    print(f"Test folder: {len(os.listdir(test_dir))} images")

if __name__ == "__main__":
    split()

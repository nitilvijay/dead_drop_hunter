import os
import shutil
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from stego_juniward import juniward_embed

# Configure logging
logging.basicConfig(
    filename='stego_jpeg_no_algo_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reset_directories(target_dir):
    print(f"Resetting {target_dir}...")
    subfolders = ["J-UNIWARD", "clean_image"]
    for folder in subfolders:
        folder_path = os.path.join(target_dir, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                shutil.move(os.path.join(folder_path, f), os.path.join(target_dir, f))
            shutil.rmtree(folder_path)

def process_single_tile(algo, source_path, dest_path, alpha):
    try:
        if algo == "J-UNIWARD":
            juniward_embed(source_path, dest_path, payload=alpha)
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise Exception("Output file missing or empty")
    except Exception as e:
        logging.error(f"FAIL: {algo} on {os.path.basename(source_path)} - {str(e)}")
        return False
    return True

def orchestrate_jpeg_steganography(target_dir, max_workers=14):
    print(f"Starting JPEG orchestration for: {target_dir}")
    reset_directories(target_dir)

    all_files = [
        f for f in os.listdir(target_dir)
        if os.path.isfile(os.path.join(target_dir, f))
        and f.endswith(".jpg")
    ]

    # Create output folders
    os.makedirs(os.path.join(target_dir, "J-UNIWARD"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "clean_image"), exist_ok=True)

    embedding_tasks = []
    alpha = 0.4

    print(f"Preparing {len(all_files)} images...")

    # Copy every image to clean_image and schedule embedding
    for filename in all_files:
        src = os.path.join(target_dir, filename)

        clean_dst = os.path.join(target_dir, "clean_image", filename)
        steg_dst = os.path.join(target_dir, "J-UNIWARD", filename)

        # Keep an untouched clean copy
        shutil.copy2(src, clean_dst)

        # Schedule embedding
        embedding_tasks.append(("J-UNIWARD", src, steg_dst, alpha))

    print(f"Embedding {len(embedding_tasks)} images...")

    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(lambda p: (p, process_single_tile(*p)), embedding_tasks)
        )

        for (algo, src, dst, _), success in results:
            if success:
                success_count += 1
            else:
                logging.error(f"Embedding failed for {os.path.basename(src)}")

    print(f"Embedding complete. Success: {success_count}/{len(embedding_tasks)}")

    # Remove originals from the root directory
    print("Removing original images...")
    for filename in all_files:
        src = os.path.join(target_dir, filename)
        if os.path.exists(src):
            os.remove(src)

    print(f"Finished processing {target_dir}")

if __name__ == "__main__":
    base_path = "/home/nitil/brainfuel/dead_drop_hunter"
    # Process train and test
    orchestrate_jpeg_steganography(os.path.join(base_path, "train_jpeg_one_algo"))
    orchestrate_jpeg_steganography(os.path.join(base_path, "test_jpeg_one_algo"))

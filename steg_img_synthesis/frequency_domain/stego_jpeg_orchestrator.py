import os
import random
import shutil
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from stego_juniward import juniward_embed
from stego_jpeg_conseal import uerd_embed, nsf5_embed

# Configure logging
logging.basicConfig(
    filename='stego_jpeg_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reset_directories(target_dir):
    print(f"Resetting {target_dir}...")
    subfolders = ["J-UNIWARD", "nsF5", "UERD", "advanced_steg", "clean_image"]
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
        elif algo == "UERD":
            uerd_embed(source_path, dest_path, alpha=alpha)
        elif algo == "nsF5":
            nsf5_embed(source_path, dest_path, alpha=alpha)
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise Exception("Output file missing or empty")
    except Exception as e:
        logging.error(f"FAIL: {algo} on {os.path.basename(source_path)} - {str(e)}")
        return False
    return True

def orchestrate_jpeg_steganography(target_dir, max_workers=12):
    print(f"Starting JPEG orchestration for: {target_dir}")
    reset_directories(target_dir)
    
    all_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)) and f.endswith(".jpg")]
    
    # Group tiles by source image
    # Filename format: {source_id}_tile{n}.jpg
    source_groups = defaultdict(list)
    for f in all_files:
        source_id = f.split('_tile')[0]
        source_groups[source_id].append(f)
    
    source_ids = list(source_groups.keys())
    random.shuffle(source_ids)
    total_sources = len(source_ids)
    
    # Distribution based on source images
    counts = {
        "J-UNIWARD": int(total_sources * 0.133),
        "nsF5": int(total_sources * 0.133),
        "UERD": int(total_sources * 0.134), # Round up to reach 40%
        "advanced_steg": int(total_sources * 0.10),
        "clean_image": int(total_sources * 0.50)
    }
    
    # Ensure remaining go to clean_image
    sum_counts = sum(counts.values())
    if sum_counts < total_sources:
        counts["clean_image"] += (total_sources - sum_counts)

    # Create folders
    for folder in counts.keys():
        os.makedirs(os.path.join(target_dir, folder), exist_ok=True)
        
    embedding_tasks = []
    current_idx = 0
    alpha = 0.4
    
    # 1. Stego Algos
    for algo in ["J-UNIWARD", "nsF5", "UERD"]:
        num = counts[algo]
        print(f"Assigning {num} source images to {algo}...")
        for _ in range(num):
            sid = source_ids[current_idx]
            tiles = source_groups[sid]
            for t in tiles:
                embedding_tasks.append((algo, os.path.join(target_dir, t), os.path.join(target_dir, algo, t), alpha))
            current_idx += 1
            
    # Execute embedding in parallel
    random.shuffle(embedding_tasks)
    print(f"Executing {len(embedding_tasks)} embedding tasks...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: (p, process_single_tile(*p)), embedding_tasks))
        
        for (algo, src, dst, _), success in results:
            if success:
                success_count += 1
                if os.path.exists(src):
                    os.remove(src)
            else:
                # Move source to clean_image folder if it fails
                shutil.move(src, os.path.join(target_dir, "clean_image", os.path.basename(src)))

    print(f"JPEG Embedding finished. Success: {success_count}/{len(embedding_tasks)}")

    # 2. Advanced Steg
    print(f"Moving {counts['advanced_steg']} source images to advanced_steg...")
    for _ in range(counts["advanced_steg"]):
        sid = source_ids[current_idx]
        for t in source_groups[sid]:
            shutil.move(os.path.join(target_dir, t), os.path.join(target_dir, "advanced_steg", t))
        current_idx += 1
        
    # 3. Clean Image
    print(f"Moving remaining source images to clean_image...")
    while current_idx < total_sources:
        sid = source_ids[current_idx]
        for t in source_groups[sid]:
            src_path = os.path.join(target_dir, t)
            if os.path.exists(src_path): # Might have been moved if stego failed
                shutil.move(src_path, os.path.join(target_dir, "clean_image", t))
        current_idx += 1

    print(f"Orchestration complete for {target_dir}.")

if __name__ == "__main__":
    base_path = "/home/nitil/brainfuel/dead_drop_hunter"
    # Process train and test
    orchestrate_jpeg_steganography(os.path.join(base_path, "train_jpeg"))
    orchestrate_jpeg_steganography(os.path.join(base_path, "test_jpeg"))

import os
import random
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from stego_lsb import lsb_embed
from stego_pvd import pvd_embed
from stego_mipod import mipod_embed
from stego_wow import wow_embed
from stego_suniward import suniward_embed

# Configure logging
logging.basicConfig(
    filename='stego_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def reset_directories(target_dir):
    print(f"Resetting {target_dir}...")
    subfolders = ["LSB", "PVD", "WOW", "S-UNIWARD", "MiPOD", "advanced_steg", "clean_image"]
    for folder in subfolders:
        folder_path = os.path.join(target_dir, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                shutil.move(os.path.join(folder_path, f), os.path.join(target_dir, f))
            shutil.rmtree(folder_path)

def process_single_image(algo, source_path, dest_path, bpp):
    try:
        if algo == "LSB":
            lsb_embed(source_path, dest_path, bpp)
        elif algo == "PVD":
            pvd_embed(source_path, dest_path, bpp)
        elif algo == "WOW":
            wow_embed(source_path, dest_path, bpp)
        elif algo == "S-UNIWARD":
            suniward_embed(source_path, dest_path, bpp)
        elif algo == "MiPOD":
            mipod_embed(source_path, dest_path, bpp)
        
        # Validation: Check if file exists and is not empty
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise Exception("Output file missing or empty")
            
    except Exception as e:
        logging.error(f"FAIL: {algo} on {os.path.basename(source_path)} - {str(e)}")
        # If it fails, we move the source to clean_image instead of leaving it in limbo
        # but the caller handles the file moves. We just signal failure.
        return False
    return True

def orchestrate_steganography(target_dir, max_workers=12):
    print(f"Starting orchestration for: {target_dir}")
    reset_directories(target_dir)
    
    all_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    random.shuffle(all_files)
    total_count = len(all_files)
    
    counts = {
        "LSB": int(total_count * 0.05),
        "PVD": int(total_count * 0.05),
        "WOW": int(total_count * 0.075),
        "S-UNIWARD": int(total_count * 0.05),
        "MiPOD": int(total_count * 0.075),
        "advanced_steg": int(total_count * 0.10),
        "clean_image": int(total_count * 0.50)
    }
    
    for folder in counts.keys():
        os.makedirs(os.path.join(target_dir, folder), exist_ok=True)
    
    tasks = []
    current_idx = 0
    bpp = 0.4

    # Prepare list of files that will be processed
    stego_assignment = [] # (algo, src, dst)

    for algo in ["LSB", "PVD", "WOW", "S-UNIWARD", "MiPOD"]:
        num = counts[algo]
        for _ in range(num):
            f = all_files[current_idx]
            stego_assignment.append((algo, os.path.join(target_dir, f), os.path.join(target_dir, algo, f), bpp))
            current_idx += 1

    # Shuffle for multi-algo parallelism
    random.shuffle(stego_assignment)

    print(f"Executing {len(stego_assignment)} embedding tasks...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: (p, process_single_image(*p)), stego_assignment))
        
        for (algo, src, dst, _), success in results:
            if success:
                success_count += 1
                # Remove source only if embedding succeeded
                if os.path.exists(src):
                    os.remove(src)
            else:
                # If failed, move source to clean_image
                shutil.move(src, os.path.join(target_dir, "clean_image", os.path.basename(src)))

    print(f"Embedding finished. Success: {success_count}/{len(stego_assignment)}. See stego_errors.log for details.")

    # Move Untouched sets
    print("Finalizing advanced_steg and clean_image sets...")
    for _ in range(counts["advanced_steg"]):
        f = all_files[current_idx]
        shutil.move(os.path.join(target_dir, f), os.path.join(target_dir, "advanced_steg", f))
        current_idx += 1

    # All remaining files go to clean_image
    for i in range(current_idx, total_count):
        f = all_files[i]
        shutil.move(os.path.join(target_dir, f), os.path.join(target_dir, "clean_image", f))

    print(f"Orchestration complete for {target_dir}.")

if __name__ == "__main__":
    base_path = "/home/nitil/brainfuel/dead_drop_hunter"
    orchestrate_steganography(os.path.join(base_path, "train_tiles"))
    orchestrate_steganography(os.path.join(base_path, "test_tiles"))

import os
import random
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor

# Import your legacy embedding backends
from steg_img_synthesis.stego_lsb import lsb_embed
from steg_img_synthesis.stego_pvd import pvd_embed
from steg_img_synthesis.stego_mipod import mipod_embed
from steg_img_synthesis.stego_wow import wow_embed
from steg_img_synthesis.stego_suniward import suniward_embed

logging.basicConfig(
    filename='stego_orchestrator.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise Exception("Output validation failed: empty or nonexistent file asset.")
        return True
    except Exception as e:
        logging.error(f"FAIL: {algo} execution on {os.path.basename(source_path)} - {str(e)}")
        return False

def orchestrate_paired_dataset(target_dir, no_img_process, max_workers=12):
    print(f"\n==========================================")
    print(f"Beginning Subfolder Balancing for: {target_dir}")
    print(f"==========================================")
    #Reads all the files in the target directory (train_tiles_1 or test_tiles_1) and shuffles them to create a random order for processing. It ensures that the number of images to process does not exceed the available files.
    all_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    random.shuffle(all_files)
    
    if len(all_files) < no_img_process:
        raise ValueError(f"Insufficient source images! Found {len(all_files)}, required exactly {no_img_process}.")
    
    #Selects the first no_img_process files from the shuffled list to create a base universe of images that will be used for embedding. This ensures that only the specified number of images are processed, and any excess files are ignored.
    base_universe = all_files[:no_img_process]
    
    cover_dir = os.path.join(target_dir, "cover")
    stego_root_dir = os.path.join(target_dir, "stego")
    
    os.makedirs(cover_dir, exist_ok=True)
    os.makedirs(stego_root_dir, exist_ok=True)
    
    allocations = [
        {"algo": "LSB", "count": no_img_process // 5},
        {"algo": "PVD", "count": no_img_process // 5},
        {"algo": "WOW", "count": no_img_process // 5},
        {"algo": "S-UNIWARD", "count": no_img_process // 5},
        {"algo": "MiPOD", "count": no_img_process // 5}
    ]
    
    embedding_tasks = [] #This is a global list that will hold tuples of (algo, source_path, dest_path, bpp) for each image that needs to be processed. Each tuple represents a single embedding task that will be executed by the worker threads.
    current_idx = 0
    bpp = 0.4
    
    # Pre-create the subfolders inside the stego root
    #Loops through the dictionary, gets algo name and the count for each algo
    for config in allocations:
        algo = config["algo"]
        algo_subfolder = os.path.join(stego_root_dir, algo)
        os.makedirs(algo_subfolder, exist_ok=True) #Creates the subfolder for each algorithm if it doesn't exist
        
        num_required = config["count"]
        print(f"Assigning {num_required} matching twins to stego/{algo}...")
        
        #Loops through the base_universe list which consits of the top 30000 image file names
        #Appends a tuple of (algo, source_path, dest_path, bpp) to the embedding_tasks list for each image that needs to be processed. 
        #The source path is the original image in the target directory, and the destination path is where the stego image will be saved in the corresponding algorithm subfolder.
        for _ in range(num_required):
            filename = base_universe[current_idx]
            src_full_path = os.path.join(target_dir, filename)
            
            # Destination path maps inside the specific algorithm nested subfolder
            dst_full_path = os.path.join(algo_subfolder, filename)
            
            embedding_tasks.append((algo, src_full_path, dst_full_path, bpp))
            current_idx += 1
            
    random.shuffle(embedding_tasks)
    #Shuffles the embedding_tasks list to randomize the order of processing, which can help distribute the workload more evenly across the worker threads.
    
    print(f"Spinning up {max_workers} worker threads to embed {no_img_process} files...")
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #Each worker picks a task from the embedding_tasks list and executes the process_single_image function.
        results = list(executor.map(lambda p: (p, process_single_image(*p)), embedding_tasks))
        
        for (algo, src, dst, _), success in results:
            if success:
                success_count += 1
            else:
                logging.warning(f"Fallback initiated for failed asset: {os.path.basename(src)}")
                lsb_embed(src, dst, bpp=0.1) 

    print(f"Embedding execution cycle complete. Successfully processed: {success_count}/{no_img_process}.")
    
    print("Populating Class 0 (Cover) directory with pristine twin profiles...")
    for filename in base_universe:
        src_loc = os.path.join(target_dir, filename)
        dst_loc = os.path.join(cover_dir, filename)
        shutil.move(src_loc, dst_loc)
        
    print("Clearing excess untiled remaining assets from core root path...")
    for filename in all_files[no_img_process:]:
        excess_file = os.path.join(target_dir, filename)
        if os.path.exists(excess_file):
            os.remove(excess_file)
            
    print(f"Orchestration pipeline finalized successfully for target partition.")

if __name__ == "__main__":
    base_path = "/home/nitil/brainfuel/dead_drop_hunter"
    
    #orchestrate_paired_dataset(os.path.join(base_path, "train_tiles_1"), no_img_process=30000, max_workers=12)
    orchestrate_paired_dataset(os.path.join(base_path, "test_tiles_1"), no_img_process=9200, max_workers=12)
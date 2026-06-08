import os
import numpy as np
from PIL import Image
import traceback

# Import our stego functions
from stego_lsb import lsb_embed
from stego_pvd import pvd_embed
from stego_mipod import mipod_embed
from stego_wow import wow_embed
from stego_suniward import suniward_embed

def run_check():
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    sample_image = os.path.join(base_dir, "train_tiles/10000_tile0.pgm")
    check_dir = os.path.join(base_dir, "check_results")
    
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    
    if not os.path.exists(sample_image):
        # Find any pgm file
        all_pgms = [f for f in os.listdir(os.path.join(base_dir, "train_tiles")) if f.endswith(".pgm")]
        if not all_pgms:
            print("No PGM tiles found in train_tiles.")
            return
        sample_image = os.path.join(base_dir, "train_tiles", all_pgms[0])
    
    print(f"Using sample image: {sample_image}")
    
    algos = [
        ("LSB", lsb_embed),
        ("PVD", pvd_embed),
        ("MiPOD", mipod_embed),
        ("WOW", wow_embed),
        ("S-UNIWARD", suniward_embed)
    ]
    
    for name, func in algos:
        print(f"\n--- Testing {name} ---")
        output_path = os.path.join(check_dir, f"sample_{name}.pgm")
        try:
            func(sample_image, output_path, bpp=0.4)
            if os.path.exists(output_path):
                print(f"SUCCESS: {name} generated {output_path}")
                # Verify it's an image
                with Image.open(output_path) as img:
                    print(f"Result Image Size: {img.size}")
            else:
                print(f"FAILED: {name} did not generate an output file.")
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_check()

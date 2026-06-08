import os
import subprocess
import shutil
import tempfile
from PIL import Image

def wow_embed(image_path, output_path, bpp=0.4):
    """
    Performs WOW embedding using the C++ binary.
    Thread-safe version using unique temporary directories.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wow_bin = os.path.join(base_dir, "WOW_linux_make_v10/executable/WOW")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # WOW binary prefers PGM.
        input_file = image_path
        temp_pgm = None
        if not image_path.lower().endswith(".pgm"):
            temp_pgm = os.path.join(temp_dir, "input.pgm")
            img = Image.open(image_path).convert("L")
            img.save(temp_pgm)
            input_file = temp_pgm

        # Output folder within the unique temp directory
        output_folder = os.path.join(temp_dir, "out")
        os.makedirs(output_folder)

        cmd = [
            wow_bin,
            "-i", input_file,
            "-O", output_folder,
            "-a", str(bpp),
            "-p", "-0.1",
            "-h", "10"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            generated_file = os.path.join(output_folder, os.path.basename(input_file))
            if os.path.exists(generated_file):
                if output_path.lower().endswith(".bmp") and generated_file.endswith(".pgm"):
                    img = Image.open(generated_file)
                    img.save(output_path)
                else:
                    shutil.copy2(generated_file, output_path)
        except subprocess.CalledProcessError as e:
            raise Exception(f"WOW binary failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stego_wow.py <input> <output> [bpp]")
    else:
        bpp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        wow_embed(sys.argv[1], sys.argv[2], bpp)

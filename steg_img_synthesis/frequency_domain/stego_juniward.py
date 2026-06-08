import os
import subprocess
import shutil
import tempfile

def juniward_embed(image_path, output_path, payload=0.4):
    """
    Performs J-UNIWARD embedding using the C++ binary.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    juniward_bin = os.path.join(base_dir, "J-UNIWARD_linux_make_v11/executable/J-UNIWARD")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "in")
        output_dir = os.path.join(temp_dir, "out")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        # Copy input file to temp input dir
        shutil.copy2(image_path, os.path.join(input_dir, os.path.basename(image_path)))
        
        # Command: J-UNIWARD -I input-dir -O output-dir -a payload -h 10
        cmd = [
            juniward_bin,
            "-I", input_dir,
            "-O", output_dir,
            "-a", str(payload),
            "-h", "10"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            generated_file = os.path.join(output_dir, os.path.basename(image_path))
            if os.path.exists(generated_file):
                shutil.copy2(generated_file, output_path)
            else:
                raise Exception("J-UNIWARD binary did not generate an output file")
        except subprocess.CalledProcessError as e:
            raise Exception(f"J-UNIWARD binary failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stego_juniward.py <input> <output> [payload]")
    else:
        payload = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        juniward_embed(sys.argv[1], sys.argv[2], payload)

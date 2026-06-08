import os
import numpy as np
from PIL import Image

def lsb_embed(image_path, output_path, bpp=0.4):
    """
    Performs standard LSB replacement.
    """
    img = Image.open(image_path).convert('L')
    data = np.array(img)
    num_pixels = data.size
    num_bits = int(num_pixels * bpp)
    
    # Generate random payload
    payload = np.random.randint(0, 2, num_bits, dtype=np.uint8)
    
    flat_data = data.flatten()
    # Apply LSB replacement
    # We replace bits in a sequential manner for simplicity
    flat_data[:num_bits] = (flat_data[:num_bits] & 254) | payload
    
    stego_img = Image.fromarray(flat_data.reshape(data.shape))
    stego_img.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stego_lsb.py <input> <output> [bpp]")
    else:
        bpp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        lsb_embed(sys.argv[1], sys.argv[2], bpp)

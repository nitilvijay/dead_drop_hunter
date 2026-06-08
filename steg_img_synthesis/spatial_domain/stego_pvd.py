import os
import numpy as np
from PIL import Image
import math

def get_pvd_range(d):
    ranges = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]
    for low, high in ranges:
        if low <= d <= high:
            return low, high
    return 128, 255

def pvd_embed(image_path, output_path, bpp=0.4):
    img = Image.open(image_path).convert('L')
    data = np.array(img, dtype=np.int16)
    rows, cols = data.shape
    
    num_pixels = data.size
    target_bits = int(num_pixels * bpp)
    
    # Random bits
    payload = np.random.randint(0, 2, target_bits + 100, dtype=np.uint8)
    bit_idx = 0
    
    stego_data = data.copy()
    
    # Process in pairs (sequential)
    for r in range(rows):
        for c in range(0, cols - 1, 2):
            if bit_idx >= target_bits:
                break
                
            p1, p2 = stego_data[r, c], stego_data[r, c+1]
            d = abs(p1 - p2)
            low, high = get_pvd_range(d)
            w = high - low + 1
            t = int(math.log2(w))
            
            if bit_idx + t > len(payload):
                break
                
            # Get t bits
            bits = payload[bit_idx : bit_idx + t]
            val = 0
            for b in bits:
                val = (val << 1) | b
            bit_idx += t
            
            new_d = low + val
            m = abs(new_d - d)
            
            if p1 >= p2 and new_d > d:
                p1 += math.ceil(m / 2)
                p2 -= math.floor(m / 2)
            elif p1 < p2 and new_d > d:
                p1 -= math.floor(m / 2)
                p2 += math.ceil(m / 2)
            elif p1 >= p2 and new_d <= d:
                p1 -= math.ceil(m / 2)
                p2 += math.floor(m / 2)
            elif p1 < p2 and new_d <= d:
                p1 += math.floor(m / 2)
                p2 -= math.ceil(m / 2)
                
            # Clipping
            p1 = np.clip(p1, 0, 255)
            p2 = np.clip(p2, 0, 255)
            
            stego_data[r, c], stego_data[r, c+1] = p1, p2
            
        if bit_idx >= target_bits:
            break
            
    stego_img = Image.fromarray(stego_data.astype(np.uint8))
    stego_img.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stego_pvd.py <input> <output> [bpp]")
    else:
        bpp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        pvd_embed(sys.argv[1], sys.argv[2], bpp)

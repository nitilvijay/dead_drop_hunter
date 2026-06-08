import os
import numpy as np
from PIL import Image
import conseal

import warnings

def mipod_embed(image_path, output_path, bpp=0.4):
    """
    Performs MiPOD embedding using the conseal package.
    """
    img = Image.open(image_path).convert('L')
    data = np.array(img, dtype=np.uint8)
    
    # Check if the image is too flat (variance is 0)
    if np.std(data) < 1e-3:
        raise ValueError("Image is too flat for MiPOD embedding")

    with warnings.catch_warnings():
        # Suppress the expected variance clipping warning for flat-ish areas
        warnings.filterwarnings("ignore", message="invalid variance in flat areas, clipping")
        # Compute MiPOD costs
        rhos = conseal.mipod.compute_cost(data)
    
    if isinstance(rhos, np.ndarray):
        rhos = (rhos, rhos)
    
    diff = conseal.simulate.ternary(
        rhos=rhos,
        alpha=bpp,
        n=data.size
    )
    
    # Add difference to cover image
    stego_data = data.astype(np.int16) + diff
    stego_data = np.clip(stego_data, 0, 255).astype(np.uint8)
    
    stego_img = Image.fromarray(stego_data)
    stego_img.save(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stego_mipod.py <input> <output> [bpp]")
    else:
        bpp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
        mipod_embed(sys.argv[1], sys.argv[2], bpp)

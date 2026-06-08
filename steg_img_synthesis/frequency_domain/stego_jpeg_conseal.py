import os
import numpy as np
import jpeglib
import conseal
import warnings

def uerd_embed(image_path, output_path, alpha=0.4):
    """
    Performs UERD embedding using conseal.
    """
    im = jpeglib.read_dct(image_path)
    rhos = conseal.uerd.compute_cost_adjusted(im.Y, im.qt[0])
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="optimization might not have converged")
        diff = conseal.simulate.ternary(
            rhos=rhos,
            alpha=alpha,
            n=im.Y.size
        )
    
    im.Y += diff.reshape(im.Y.shape).astype(im.Y.dtype)
    im.write_dct(output_path)

def nsf5_embed(image_path, output_path, alpha=0.4):
    """
    Performs nsF5 embedding using conseal.
    """
    im = jpeglib.read_dct(image_path)
    rhos = conseal.nsF5.compute_cost_adjusted(im.Y)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="optimization might not have converged")
        diff = conseal.simulate.binary(
            rhos=rhos,
            alpha=alpha,
            n=im.Y.size
        )
    
    im.Y += diff.reshape(im.Y.shape).astype(im.Y.dtype)
    im.write_dct(output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python stego_jpeg_conseal.py <algo> <input> <output> [alpha]")
    else:
        algo = sys.argv[1].lower()
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
        
        if algo == "uerd":
            uerd_embed(input_file, output_file, alpha)
        elif algo == "nsf5":
            nsf5_embed(input_file, output_file, alpha)
        else:
            print(f"Unknown algo: {algo}")

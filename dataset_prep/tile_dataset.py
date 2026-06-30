import os
from PIL import Image

def tile_images(source_dir, output_dir, tile_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.pgm', '.bmp', '.jpg', '.png'))]
    total_files = len(files)
    print(f"Found {total_files} images in {source_dir} to tile.")

    for i, filename in enumerate(files):
        source_path = os.path.join(source_dir, filename)
        base_name, ext = os.path.splitext(filename)

        try:
            with Image.open(source_path) as img:
                width, height = img.size
                
                # Calculate number of tiles in each dimension
                nx = width // tile_size
                ny = height // tile_size
                
                tile_count = 0
                for row in range(ny):
                    for col in range(nx):
                        left = col * tile_size
                        top = row * tile_size
                        right = left + tile_size
                        bottom = top + tile_size
                        
                        # Crop and save tile
                        tile = img.crop((left, top, right, bottom))
                        tile_filename = f"{base_name}_tile{tile_count}{ext}"
                        tile.save(os.path.join(output_dir, tile_filename))
                        tile_count += 1
            
            if (i + 1) % 500 == 0 or (i + 1) == total_files:
                print(f"Processed {i + 1}/{total_files} images from {source_dir}.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    base_dir = "/home/nitil/brainfuel/dead_drop_hunter"
    
    # Tile Train set
    tile_images(os.path.join(base_dir, "train"), os.path.join(base_dir, "train_tiles"))
    
    # Tile Test set
    tile_images(os.path.join(base_dir, "test"), os.path.join(base_dir, "test_tiles"))
    
    print("Tiling complete.")

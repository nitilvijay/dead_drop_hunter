import os
from PIL import Image

def convert_images_to_grayscale(source_dir, dest_dir):
    """
    Converts all images in source_dir to grayscale and saves them to dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    total_files = len(files)
    print(f"Found {total_files} images to convert.")

    for i, filename in enumerate(files):
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        try:
            with Image.open(source_path) as img:
                grayscale_img = img.convert('L')
                grayscale_img.save(dest_path)
            
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                print(f"Converted {i + 1}/{total_files} images.")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    source_directory = "/home/nitil/brainfuel/dead_drop_hunter/sp4g8h7v8k-1/NRC-BMP-1500"
    destination_directory = "/home/nitil/brainfuel/dead_drop_hunter/sp4g8h7v8k-1/NRC-BMP-1500-grayscale"
    
    convert_images_to_grayscale(source_directory, destination_directory)
    print("Conversion complete.")

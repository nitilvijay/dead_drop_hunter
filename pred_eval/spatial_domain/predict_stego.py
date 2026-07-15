import argparse
import os
import time
import numpy as np
import torch
from PIL import Image

# Assuming BASE_DIR and XuNet are imported from your custom package
from nvidia_gpu_train import BASE_DIR, XuNet

# Default checkpoint for single-model inference
DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "epoch_11.pth")
VALID_EXTS = (".pgm", ".bmp", ".png", ".jpg", ".jpeg")


def select_prediction_device(requested):
    req = requested.lower()
    if req == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print("[Warning] Intel XPU requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("xpu")

    if req == "cpu":
        return torch.device("cpu")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """Load a single model checkpoint for inference."""
    print(f"Loading checkpoint '{os.path.basename(checkpoint_path)}' into {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = XuNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image_tensor(image_path):
    print(image_path)
    with Image.open(image_path) as img:
        arr = np.array(img, dtype=np.float32)

    if arr.ndim == 2:
        tensor = torch.from_numpy(arr)
    elif arr.ndim == 3:
        tensor = torch.from_numpy(arr[:, :, 0])
    else:
        raise ValueError(f"Corrupted image dimensions {arr.shape} at {image_path}")

    if tensor.shape != (256, 256):
        raise ValueError(f"Expected 256x256, got {tensor.shape} at {image_path}")

    return tensor.unsqueeze(0).unsqueeze(0)


def predict_single_model_batch(model, tensor_batch, device):
    """Run a single model on a batch and return class probabilities."""
    inputs = tensor_batch.to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    return probs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forensic Scanner: 3-Model Ensemble Steganography Detection."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Path to a single image OR a directory of images.",
    )
    # Accepts 3 checkpoint paths separated by spaces
    parser.add_argument(
        "--checkpoint",
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help="Path to a model checkpoint file to use for inference."
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        choices=["auto", "xpu", "cpu"],
        help="Hardware backend to execute inference on."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.50,
        help="Ensemble probability threshold to flag STEGO (default: 0.50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_path = args.target or input("Enter image or directory path: ").strip("'")
    target_path = os.path.abspath(os.path.expanduser(target_path))
    
    # Resolve single checkpoint path
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target not found: {target_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = select_prediction_device(args.device)
    
    # Load the single model upfront
    model = load_model(checkpoint_path, device)

    # Gather files
    files_to_scan = []
    if os.path.isfile(target_path):
        files_to_scan.append(target_path)
    else:
        for root, _, files in os.walk(target_path):
            for f in files:
                if f.lower().endswith(VALID_EXTS):
                    files_to_scan.append(os.path.join(root, f))
        files_to_scan.sort()

    if not files_to_scan:
        print(f"No valid images found in {target_path}")
        return

    print(f"\nScanning {len(files_to_scan)} file(s) using single model (threshold >= {args.threshold * 100:.1f}%)...\n")
    print(f"{'Filename':<40} | {'Result':<8} | {'Stego Prob':<15} | {'Time'}")
    print("-" * 80)

    stego_count = 0
    start_time = time.perf_counter()

    for file_path in files_to_scan:
        try:
            t0 = time.perf_counter()
            img_tensor = load_image_tensor(file_path)

            probs = predict_single_model_batch(model, img_tensor, device)[0]
            elapsed = (time.perf_counter() - t0) * 1000

            clean_prob, stego_prob = probs[0], probs[1]
            is_stego = stego_prob >= args.threshold
            verdict = "STEGO" if is_stego else "CLEAN"
            
            if is_stego:
                stego_count += 1

            disp_name = os.path.basename(file_path)
            if len(disp_name) > 37:
                disp_name = disp_name[:34] + "..."

            print(f"{disp_name:<40} | {verdict:<8} | {stego_prob * 100:6.2f}%         | {elapsed:5.1f}ms")

        except Exception as e:
            print(f"{os.path.basename(file_path):<40} | ERROR    | {str(e)}")

    total_time = time.perf_counter() - start_time
    print("-" * 80)
    print(f"Scan Complete: {stego_count} STEGO / {len(files_to_scan) - stego_count} CLEAN detected in {total_time:.2f}s")


if __name__ == "__main__":
    main()
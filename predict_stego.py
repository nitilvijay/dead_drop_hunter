import argparse
import os

import numpy as np
import torch
from PIL import Image

from steg_train import BASE_DIR, XuNet, select_device


DEFAULT_CHECKPOINT = os.path.join(
    BASE_DIR, "checkpoints", "xunet_epoch_8.pth"
)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = XuNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    with Image.open(image_path) as image:
        array = np.array(image.convert("L"), dtype=np.float32)

    if array.shape != (256, 256):
        raise ValueError(
            f"Expected a 256x256 image, but received {array.shape}."
        )

    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def predict(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    clean_probability = probabilities[0].item()
    stego_probability = probabilities[1].item()
    prediction = "STEGO" if stego_probability >= clean_probability else "CLEAN"
    return prediction, clean_probability, stego_probability


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether a 256x256 image is clean or stego."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Image path. You will be prompted when this is omitted.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=["auto", "xpu", "cpu"], default="auto"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image or input("Enter the image file path: ").strip()
    image_path = os.path.abspath(os.path.expanduser(image_path))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = select_device(args.device)
    model = load_model(checkpoint_path, device)
    image_tensor = load_image(image_path)
    prediction, clean_probability, stego_probability = predict(
        model, image_tensor, device
    )

    print(f"Device: {device}")
    print(f"Prediction: {prediction}")
    print(f"Clean probability: {clean_probability * 100:.2f}%")
    print(f"Stego probability: {stego_probability * 100:.2f}%")


if __name__ == "__main__":
    main()

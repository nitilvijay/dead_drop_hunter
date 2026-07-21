import io
import os
import traceback
import urllib.request
import uuid

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from nvidia_gpu_train import XuNet


app = FastAPI(title="Steganography Detector")
device = torch.device("cpu")

def normalize_ec2_url(raw_url):
    url = raw_url.rstrip("/")
    for suffix in ["/api/store_clean","/api/store_normal","/api/store_heatmap"]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


EC2_BACKEND_URL = normalize_ec2_url(os.environ["EC2_BACKEND_URL"])
EC2_CLEAN_URL = f"{EC2_BACKEND_URL}/api/store_clean"
EC2_NORMAL_URL = f"{EC2_BACKEND_URL}/api/store_normal"
EC2_HEATMAP_URL = f"{EC2_BACKEND_URL}/api/store_heatmap"
TILE_SIZE = 256


def load_model():
    model = XuNet()
    checkpoint = torch.load("epoch_13.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_grayscale_array(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(image, dtype=np.float32)


def pad_to_tile_size(image_array):
    height, width = image_array.shape
    padded_height = ((height + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
    padded_width = ((width + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    if padded_height == height and padded_width == width:
        return image_array, height, width

    padded_image = np.pad(
        image_array,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    return padded_image, height, width


def predict_tile_probabilities(model, padded_image):
    probabilities = []
    rows = padded_image.shape[0] // TILE_SIZE
    cols = padded_image.shape[1] // TILE_SIZE

    for row in range(rows):
        for col in range(cols):
            top = row * TILE_SIZE
            left = col * TILE_SIZE
            tile = padded_image[top : top + TILE_SIZE, left : left + TILE_SIZE]
            tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            probabilities.append(float(probs[1]))

    return probabilities, rows, cols


def build_heatmap_image(padded_image, probabilities, rows, cols, original_height, original_width):
    prob_grid = np.array(probabilities, dtype=np.float32).reshape(rows, cols)

    heatmap_rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            value = prob_grid[row, col]
            heatmap_rgb[row, col] = [int(255 * value), 0, int(255 * (1 - value))]

    heatmap_resized = np.repeat(np.repeat(heatmap_rgb, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
    original_rgb = np.stack((padded_image.astype(np.uint8),) * 3, axis=-1)
    blended = (original_rgb * 0.5 + heatmap_resized * 0.5).astype(np.uint8)
    blended = blended[:original_height, :original_width]
    return Image.fromarray(blended)


def image_to_png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_multipart_form(file_name, content_type, content_bytes):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()

    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode())
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.write(content_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    return body.getvalue(), boundary


def post_file(url, file_name, content_type, content_bytes):
    payload, boundary = build_multipart_form(file_name, content_type, content_bytes)
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        return response.read()


@app.post("/api/detect")
async def detect_steganography(file: UploadFile = File(...)):
    print("[Info] Received file for detection:", file.filename)
    try:
        image_bytes = await file.read()
        image_array = load_grayscale_array(image_bytes)
        padded_image, original_height, original_width = pad_to_tile_size(image_array)

        model = load_model()
        probabilities, rows, cols = predict_tile_probabilities(model, padded_image)

        stego_tiles = sum(probability > 0.5 for probability in probabilities)
        total_tiles = len(probabilities)
        stego_ratio = stego_tiles / total_tiles if total_tiles else 0.0
        any_high_tile = any(probability > 0.7 for probability in probabilities)
        is_stego = stego_ratio > 0.3 or any_high_tile

        print(
            f"[Info] File: {file.filename}, Tiles: {total_tiles}, Stego Tiles: {stego_tiles}, "
            f"Stego Ratio: {stego_ratio:.4f}, Any Tile > 0.7: {any_high_tile}, Verdict: {'STEGO' if is_stego else 'CLEAN'}"
        )

        if not is_stego:
            clean_name = file.filename or "image"

            print(f"[Info] Forwarding clean image to: {EC2_CLEAN_URL}")

            try:
                post_file(
                    EC2_CLEAN_URL,
                    clean_name,
                    file.content_type or "application/octet-stream",
                    image_bytes,
                )

            except Exception:
                print(f"[Error] Failed forwarding clean image to: {EC2_CLEAN_URL}")
                traceback.print_exc()
                raise

            return {
                "success": True,
                "message": "Stored as clean.",
                "verdict": "CLEAN",
            }

        heatmap_image = build_heatmap_image(
            padded_image,
            probabilities,
            rows,
            cols,
            original_height,
            original_width,
        )

        normal_name = file.filename or "image"
        heatmap_name = f"{os.path.splitext(normal_name)[0]}_heatmap.png"

        print(f"[Info] Forwarding normal image to: {EC2_NORMAL_URL}")
        print(f"[Info] Forwarding heatmap image to: {EC2_HEATMAP_URL}")
        try:
            post_file(
                EC2_NORMAL_URL,
                normal_name,
                file.content_type or "application/octet-stream",
                image_bytes,
            )
            post_file(
                EC2_HEATMAP_URL,
                heatmap_name,
                "image/png",
                image_to_png_bytes(heatmap_image),
            )
        except Exception:
            print(f"[Error] Failed forwarding to EC2 backend. Normal URL: {EC2_NORMAL_URL}")
            print(f"[Error] Failed forwarding to EC2 backend. Heatmap URL: {EC2_HEATMAP_URL}")
            traceback.print_exc()
            raise

        return {
            "success": True,
            "message": "Stored in quarantine.",
            "verdict": "STEGO",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("hf_backend:app", host="0.0.0.0", port=8000, workers=1, reload=True)
import io
import os
import uuid

import boto3
import botocore
import uvicorn
from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Image Storage Backend")

# S3
s3 = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")

# Local fallback directories
QUARANTINE_DIR = os.getenv("QUARANTINE_DIR", "quarantine")
QUARANTINE_NORMAL_DIR = os.getenv("QUARANTINE_NORMAL_DIR", "quarantine_normal")
CLEAN_DIR = os.getenv("CLEAN_DIR", "clean")

os.makedirs(QUARANTINE_DIR, exist_ok=True)
os.makedirs(QUARANTINE_NORMAL_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)


def save_upload_bytes(directory: str, image_bytes: bytes, filename: str):
    file_path = os.path.join(directory, filename)

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved locally -> {file_path}")


def save_to_s3(image_bytes: bytes, filename: str) -> bool:
    """
    Returns True if upload succeeded.
    Returns False if S3 could not be reached.
    """
    try:
        s3.upload_fileobj(
            io.BytesIO(image_bytes),
            S3_BUCKET,
            filename,
        )

        print(f"Uploaded to S3 -> {filename}")
        return True

    except (
        botocore.exceptions.EndpointConnectionError,
        botocore.exceptions.NoCredentialsError,
        botocore.exceptions.ClientError,
        botocore.exceptions.BotoCoreError,
    ) as e:

        print("S3 bucket not reachable.")
        print(e)
        return False


@app.post("/api/store_clean")
async def store_clean(file: UploadFile = File(...)):
    print("[Info] Received file for clean storage:", file.filename)
    image_bytes = await file.read()
    filename = file.filename or f"{uuid.uuid4().hex}.png"

    uploaded = save_to_s3(image_bytes, filename)

    if not uploaded:
        save_upload_bytes(CLEAN_DIR, image_bytes, filename)

    return {
        "filename": filename,
        "stored": "s3" if uploaded else "local"
    }


@app.post("/api/store_normal")
async def store_normal(file: UploadFile = File(...)):
    print("[Info] Received file for normal storage:", file.filename)
    image_bytes = await file.read()
    filename = file.filename or f"{uuid.uuid4().hex}.png"

    save_upload_bytes(QUARANTINE_NORMAL_DIR, image_bytes, filename)

    return {
        "filename": filename
    }


@app.post("/api/store_heatmap")
async def store_heatmap(file: UploadFile = File(...)):
    print("[Info] Received file for heatmap storage:", file.filename)
    image_bytes = await file.read()
    filename = file.filename or f"{uuid.uuid4().hex}.png"

    save_upload_bytes(QUARANTINE_DIR, image_bytes, filename)

    return {
        "filename": filename
    }


if __name__ == "__main__":
    uvicorn.run(
        "ec2_backend:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )
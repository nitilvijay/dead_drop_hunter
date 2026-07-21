import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, f1_score
import glob
from tqdm import tqdm
import jpeglib

BASE_DIR = "/home/nitil/brainfuel/dead_drop_hunter"
STEGO_TYPES = ["J-UNIWARD"]

def select_device(requested):
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "XPU was requested, but PyTorch cannot access an Intel GPU. "
                "Check the Intel GPU driver and XPU runtime installation."
            )
        return torch.device("xpu")

    if requested == "cpu":
        return torch.device("cpu")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

# 1. Grid-Strided Unfolding Layer
class PhaseUnfoldingLayer(nn.Module):
    def __init__(self):
        super(PhaseUnfoldingLayer, self).__init__()

    def forward(self, x):
        # x is [B, 1, 256, 256]
        B, C, H, W = x.size()
        
        # We need to extract 64 channels based on 8x8 block positions
        # Reshape to [B, C, 32, 8, 32, 8]
        x = x.view(B, C, H // 8, 8, W // 8, 8)
        
        # Transpose to group the 8x8 positions together: [B, C, 8, 8, 32, 32]
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        
        # Flatten the 8x8 block positions into 64 channels: [B, C * 64, 32, 32]
        # Since C=1, this becomes [B, 64, 32, 32]
        x = x.view(B, C * 64, H // 8, W // 8)
        
        # Apply Hard Constraints
        x = torch.clamp(x, min=-2.0, max=2.0)
        x = torch.abs(x)
        
        return x

# 2. Global Covariance Pooling Layer
class GlobalCovariancePooling(nn.Module):
    def __init__(self, eps=1e-5):
        super(GlobalCovariancePooling, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        # Reshape to [B, C, H*W]
        x = x.view(B, C, N)

        # Compute mean along the spatial dimensions
        mean = x.mean(dim=2, keepdim=True)
        x_hat = x - mean

        # Compute Covariance: (X * X^T) / (N - 1)
        # Resulting shape: [B, C, C]
        cov = torch.bmm(x_hat, x_hat.transpose(1, 2)) / (N - 1)

        # Add a tiny epsilon to the diagonal for numerical stability
        iden = torch.eye(C, device=x.device).view(1, C, C).repeat(B, 1, 1)
        cov = cov + self.eps * iden

        return cov


class JpegStegModel(nn.Module):
    def __init__(self):
        super(JpegStegModel, self).__init__()
        self.unfold = PhaseUnfoldingLayer()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Reduce 512 channels down to 64 to avoid a massive 512x512 covariance matrix
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Global Covariance Pooling outputs a 64x64 matrix per sample
        self.global_cov_pool = GlobalCovariancePooling()

        # Output Head: Fully Connected (64 * 64 = 4096 inputs)
        self.fc = nn.Linear(64 * 64, 2)

    def forward(self, x):
        x = self.unfold(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_reduce(x)
        x = self.global_cov_pool(x)
        x = x.view(x.size(0), -1) # Flatten the covariance matrix
        x = self.fc(x)
        return x



# 3. Dataset and Dataloader
class JpegStegoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        samples_per_stego_type=None,
        clean_samples=None,
    ):
        self.root_dir = root_dir
        self.stego_types = STEGO_TYPES
        rng = np.random.default_rng()
        
        # Gather all stego files
        self.stego_files = []
        self.stego_categories = []
        for stype in self.stego_types:
            path = os.path.join(root_dir, stype, "*.*")
            found = sorted(glob.glob(path))
            if samples_per_stego_type and len(found) > samples_per_stego_type:
                indices = rng.choice(
                    len(found), samples_per_stego_type, replace=False
                )
                found = [found[i] for i in indices]
            self.stego_files.extend(found)
            self.stego_categories.extend([stype] * len(found))
            
        # Gather clean files
        self.clean_files = sorted(glob.glob(os.path.join(root_dir, "clean_image", "*.*")))
        if clean_samples and len(self.clean_files) > clean_samples:
            indices = rng.choice(len(self.clean_files), clean_samples, replace=False)
            self.clean_files = [self.clean_files[i] for i in indices]
        
        # Balance the dataset 50/50
        num_stego = len(self.stego_files)
        num_clean = len(self.clean_files)
        
        print(f"[{split}] Found {num_stego} Stego and {num_clean} Clean images.")

        if num_stego == 0 or num_clean == 0:
            raise ValueError(
                f"{root_dir} must contain both stego and clean images; "
                f"found {num_stego} stego and {num_clean} clean."
            )
        
        if num_clean > num_stego:
            # Undersample clean
            indices = rng.choice(len(self.clean_files), num_stego, replace=False)
            self.clean_files = [self.clean_files[i] for i in indices]
        elif num_stego > num_clean:
            # Undersample stego
            indices = rng.choice(len(self.stego_files), num_clean, replace=False)
            self.stego_files = [self.stego_files[i] for i in indices]
            self.stego_categories = [self.stego_categories[i] for i in indices]
            
        self.files = self.stego_files + self.clean_files
        self.labels = [1] * len(self.stego_files) + [0] * len(self.clean_files)
        self.categories = self.stego_categories + ["clean"] * len(self.clean_files)
        
        print(f"[{split}] Final balanced dataset size: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        # Read the REAL DCT coefficients using jpeglib
        jpeg_struct = jpeglib.read_dct(img_path)

        # We use the Y (Luminance) channel for steganalysis
        dct_y = jpeg_struct.Y

        if dct_y.shape != (32, 32, 8, 8):
            raise ValueError(f"Expected a 32x32x8x8 DCT structure, got {dct_y.shape}: {img_path}")

        # Reshape to a flat spatial representation [256, 256] 
        # so the PhaseUnfoldingLayer can process it correctly
        # The unfolding layer expects [1, 256, 256] as input
        dct_spatial = np.zeros((256, 256), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                dct_spatial[i*8:(i+1)*8, j*8:(j+1)*8] = dct_y[i, j, :, :]

        tensor = torch.from_numpy(dct_spatial).unsqueeze(0)
        label = self.labels[idx]
        return tensor, label, self.categories[idx]


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy, args):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_balanced_accuracy": best_accuracy,
        "config": vars(args),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        checkpoint["xpu_random_state"] = torch.xpu.get_rng_state_all()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if device.type == "xpu" and "xpu_random_state" in checkpoint:
        torch.xpu.set_rng_state_all(checkpoint["xpu_random_state"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_balanced_accuracy", 0.0)


def report_metrics(labels, predictions, categories):
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    clean_accuracy = matrix[0, 0] / matrix[0].sum() if matrix[0].sum() else 0.0
    stego_recall = matrix[1, 1] / matrix[1].sum() if matrix[1].sum() else 0.0

    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"F1 Score:          {f1 * 100:.2f}%")
    print(f"Clean accuracy:    {clean_accuracy * 100:.2f}%")
    print(f"Stego recall:      {stego_recall * 100:.2f}%")

    for category in STEGO_TYPES:
        category_indices = [i for i, value in enumerate(categories) if value == category]
        if category_indices:
            correct = sum(predictions[i] == 1 for i in category_indices)
            accuracy = correct / len(category_indices)
            print(f"{category:16s} recall: {accuracy * 100:.2f}% ({correct}/{len(category_indices)})")

    return balanced_accuracy


def train_model(args):
    device = select_device(args.device)
    device_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    print(f"Using device: {device} ({device_name})")
    
    # Load Datasets
    train_dataset = JpegStegoDataset(
        os.path.join(BASE_DIR, "train_jpeg_one_algo"),
        split="train",
        samples_per_stego_type=args.train_samples_per_type,
        clean_samples=args.train_clean_samples,
    )
    test_dataset = JpegStegoDataset(
        os.path.join(BASE_DIR, "test_jpeg_one_algo"),
        split="test",
        samples_per_stego_type=args.test_samples_per_type,
        clean_samples=args.test_clean_samples,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    
    # Initialize Model
    model = JpegStegModel().to(device)
    
    # Weight decay added for L2 Regularization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        start_epoch, best_accuracy = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(f"Resumed from epoch {start_epoch}")

    checkpoint_dir = os.path.join(BASE_DIR, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training loop with progress bar
        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels, _ in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        scheduler.step()
        
        # Evaluation with progress bar
        model.eval()
        all_preds = []
        all_labels = []
        all_categories = []
        
        eval_bar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for inputs, labels, categories in eval_bar:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_categories.extend(categories)
        
        balanced_accuracy = report_metrics(all_labels, all_preds, all_categories)
        best_accuracy = max(best_accuracy, balanced_accuracy)

        checkpoint_path = os.path.join(
            checkpoint_dir, f"jpegsteg_epoch_{epoch + 1}.pth"
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_accuracy,
            args,
        )
        print(f"Saved checkpoint: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train JpegSteg model")
    parser.add_argument("--device", choices=["auto", "xpu", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--checkpoint-dir", default="checkpoints_jpeg_1")
    parser.add_argument("--train-samples-per-type", type=int)
    parser.add_argument("--train-clean-samples", type=int)
    parser.add_argument("--test-samples-per-type", type=int)
    parser.add_argument("--test-clean-samples", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    train_model(parse_args())
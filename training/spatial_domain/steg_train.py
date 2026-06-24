import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, f1_score
from tqdm import tqdm

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

# 1. SRM Layer Definition
class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        kernels = np.load(os.path.join(BASE_DIR, "SRM_Kernels1.npy"))
        kernels = np.transpose(kernels, (3, 2, 0, 1))
        
        self.conv = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight.data = torch.from_numpy(kernels).float()
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return torch.clamp(self.conv(x), min=-2.0, max=2.0)


class GlobalCovariancePooling(nn.Module):
    def __init__(self, eps=1e-8):
        super(GlobalCovariancePooling, self).__init__()
        self.eps = eps

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        features = x.reshape(batch_size, channels, -1)
        features = features - features.mean(dim=2, keepdim=True)

        denom = max(features.size(2) - 1, 1)
        covariance = torch.bmm(features, features.transpose(1, 2)) / denom

        upper_indices = torch.triu_indices(channels, channels, device=x.device)
        covariance = covariance[:, upper_indices[0], upper_indices[1]]
        covariance = torch.sign(covariance) * torch.sqrt(torch.abs(covariance) + self.eps)
        return F.normalize(covariance, p=2, dim=1)


# 2. Xu-Net Architecture
class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        self.srm = SRMLayer()
        
        self.layer1 = nn.Sequential(nn.Conv2d(30, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.Tanh())
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.Tanh())
        
        self.global_pool = GlobalCovariancePooling()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * (256 + 1) // 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_pool(x)
        return self.fc(x)


# 3. Dataset (Shape safety check preserved)
class StegoDataset(Dataset):
    def __init__(self, root_dir, split="train", samples_per_stego_type=None, clean_samples=None):
        self.root_dir = root_dir
        self.stego_types = STEGO_TYPES
        rng = np.random.default_rng()
        
        self.stego_files = []
        self.stego_categories = []
        for stype in self.stego_types:
            path = os.path.join(root_dir, stype, "*.*")
            found = sorted(glob.glob(path))
            if samples_per_stego_type and len(found) > samples_per_stego_type:
                indices = rng.choice(len(found), samples_per_stego_type, replace=False)
                found = [found[i] for i in indices]
            self.stego_files.extend(found)
            self.stego_categories.extend([stype] * len(found))
            
        self.clean_files = sorted(glob.glob(os.path.join(root_dir, "clean_image", "*.*")))
        if clean_samples and len(self.clean_files) > clean_samples:
            indices = rng.choice(len(self.clean_files), clean_samples, replace=False)
            self.clean_files = [self.clean_files[i] for i in indices]
        
        num_stego = len(self.stego_files)
        num_clean = len(self.clean_files)
        
        print(f"[{split}] Found {num_stego} Stego and {num_clean} Clean images.")
        if num_stego == 0 or num_clean == 0:
            raise ValueError(f"Found {num_stego} stego and {num_clean} clean images. Both required.")
        
        if num_clean > num_stego:
            indices = rng.choice(len(self.clean_files), num_stego, replace=False)
            self.clean_files = [self.clean_files[i] for i in indices]
        elif num_stego > num_clean:
            indices = rng.choice(len(self.stego_files), num_clean, replace=False)
            self.stego_files = [self.stego_files[i] for i in indices]
            self.stego_categories = [self.stego_categories[i] for i in indices]
            
        self.files = self.stego_files + self.clean_files
        self.labels = [1] * len(self.stego_files) + [0] * len(self.clean_files)
        self.categories = self.stego_categories + ["clean"] * len(self.clean_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        with Image.open(img_path) as img:
            arr = np.array(img.convert("L"), dtype=np.float32)

        if arr.shape != (256, 256):
            raise ValueError(f"Corrupted image shape {arr.shape} at: {img_path}")

        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, self.labels[idx], self.categories[idx]


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_balanced_accuracy": best_accuracy,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_balanced_accuracy", 0.0)


def report_metrics(labels, predictions, categories):
    bal_acc = balanced_accuracy_score(labels, predictions)
    print(f"Balanced accuracy: {bal_acc * 100:.2f}%")
    print(f"Precision:         {precision_score(labels, predictions) * 100:.2f}%")
    print(f"F1 Score:          {f1_score(labels, predictions) * 100:.2f}%")
    
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    print(f"Clean accuracy:    {(cm[0, 0] / cm[0].sum()) * 100:.2f}%")
    print(f"Stego recall:      {(cm[1, 1] / cm[1].sum()) * 100:.2f}%")

    for cat in STEGO_TYPES:
        cat_idx = [i for i, v in enumerate(categories) if v == cat]
        if cat_idx:
            correct = sum(predictions[i] == 1 for i in cat_idx)
            print(f"{cat:15s} recall: {(correct / len(cat_idx)) * 100:.2f}%")
    return bal_acc


# ---------------- INTEL TRAINING ENGINE ----------------
def train_model(config):
    device = select_device()
    device_name = torch.xpu.get_device_name(0)
    print(f"🚀 Bound to Intel Hardware: {device} ({device_name})")
    
    train_dataset = StegoDataset(
        config["train_dir"], split="train",
        samples_per_stego_type=config["train_samples_per_type"],
        clean_samples=config["train_clean_samples"]
    )
    test_dataset = StegoDataset(
        config["test_dir"], split="test",
        samples_per_stego_type=config["test_samples_per_type"],
        clean_samples=config["test_clean_samples"]
    )
    
    # VIP memory pinning active
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["workers"], pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["workers"], pin_memory=True, persistent_workers=True
    )
    
    model = XuNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # --- INTEL SPEED HACK: Hardware Graph Compilation ---
    if HAS_IPEX:
        print("⚡ Fusing network layers via ipex.optimize()...")
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)

    start_epoch, best_accuracy = 0, 0.0
    if config["resume"]:
        start_epoch, best_accuracy = load_checkpoint(config["resume"], model, optimizer, scheduler, device)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels, _ in train_bar:
            
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- INTEL SPEED HACK: Native BrainFloat16 Autocasting ---
            # Pure 16-bit wide range math. Zero GradScalers required.
            with torch.amp.autocast(device_type="xpu", dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        scheduler.step()
        
        # --- Evaluation ---
        model.eval()
        all_preds, all_labels, all_categories = [], [], []
        
        eval_bar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for inputs, labels, categories in eval_bar:
                inputs = inputs.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="xpu", dtype=torch.bfloat16):
                    outputs = model(inputs)
                    
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_categories.extend(categories)
        
        balanced_accuracy = report_metrics(all_labels, all_preds, all_categories)
        best_accuracy = max(best_accuracy, balanced_accuracy)

        ckpt_path = os.path.join(config["checkpoint_dir"], f"xunet_epoch_{epoch + 1}.pth")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_accuracy, config)
        print(f"Saved secure XPU checkpoint -> {ckpt_path}")


# Execution Config (Update the directory paths to match your Intel workstation!)
config = {
    "epochs": 10,
    "batch_size": 32,
    "workers": 4,  # Intel workstations usually have 8-32 CPU cores; 4 is great here.
    "learning_rate": 0.0008,
    "resume": None, 

    # CHANGE THESE 3 DIRECTORIES TO YOUR LOCAL MACHINE'S ACTUAL FOLDERS:
    "train_dir": "./train_tiles",
    "test_dir": "./test_tiles",
    "checkpoint_dir": "./checkpoints", 

    "train_samples_per_type": None,
    "train_clean_samples": None,
    "test_samples_per_type": None,
    "test_clean_samples": None
}

if __name__ == "__main__":
    train_model(config)
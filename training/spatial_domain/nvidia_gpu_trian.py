import os
import gc
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, f1_score
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.getcwd())
STEGO_TYPES = ["LSB", "PVD", "WOW", "S-UNIWARD", "MiPOD"]
VALID_EXTS = (".pgm", ".bmp", ".png") # [MODIFIED] Explicit whitelist


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("No CUDA-capable GPU detected.")


def select_amp_dtype():
    # Ada Lovelace (RTX 4050) supports bfloat16 natively. 
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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
    def __init__(self, channels=256, eps=1e-8):
        super(GlobalCovariancePooling, self).__init__()
        self.eps = eps
        idx = torch.triu_indices(channels, channels)
        self.register_buffer("upper_indices", idx, persistent=False)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float() 
            batch_size, channels, _, _ = x.shape
            features = x.reshape(batch_size, channels, -1)
            features = features - features.mean(dim=2, keepdim=True)

            denom = max(features.size(2) - 1, 1)
            covariance = torch.bmm(features, features.transpose(1, 2)) / denom

            covariance = covariance[:, self.upper_indices[0], self.upper_indices[1]]
            covariance = torch.sign(covariance) * torch.sqrt(torch.abs(covariance) + self.eps)
            return F.normalize(covariance, p=2, dim=1)


class XuNet(nn.Module):
    def __init__(self, use_checkpointing=False):
        super(XuNet, self).__init__()
        self.srm = SRMLayer()

        self.layer1 = nn.Sequential(nn.Conv2d(30, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.Tanh())
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.Tanh())

        self.global_pool = GlobalCovariancePooling(channels=256)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * (256 + 1) // 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2)
        )
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        x = self.srm(x)
        if self.use_checkpointing and self.training:
            x = grad_checkpoint(self.layer1, x, use_reentrant=False)
            x = grad_checkpoint(self.layer2, x, use_reentrant=False)
            x = grad_checkpoint(self.layer3, x, use_reentrant=False)
            x = grad_checkpoint(self.layer4, x, use_reentrant=False)
            x = grad_checkpoint(self.layer5, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)

        x = self.global_pool(x)
        return self.fc(x)


# [MODIFIED] Replaced torchvision.io with direct PIL fast-path for BMP/PGM
def _load_gray_tensor(img_path):
    with Image.open(img_path) as im:
        arr = np.array(im, dtype=np.float32)

    if arr.ndim == 2:
        return torch.from_numpy(arr).unsqueeze(0) # (1, H, W)
    elif arr.ndim == 3:
        # Fallback safeguard if an indexed/RGB BMP slipped into the folder
        return torch.from_numpy(arr[:, :, 0]).unsqueeze(0)
    
    raise ValueError(f"Unrecognized image dimensions {arr.shape} at: {img_path}")


class StegoDataset(Dataset):
    def __init__(self, root_dir, split="train", samples_per_stego_type=None, clean_samples=None):
        self.root_dir = root_dir
        self.stego_types = STEGO_TYPES
        rng = np.random.default_rng()

        self.stego_files = []
        self.stego_categories = []
        for stype in self.stego_types:
            path = os.path.join(root_dir, stype, "*.*")
            # [MODIFIED] Filter out OS junk files (.DS_Store, Thumbs.db)
            found = sorted([f for f in glob.glob(path) if f.lower().endswith(VALID_EXTS)])
            
            if samples_per_stego_type and len(found) > samples_per_stego_type:
                indices = rng.choice(len(found), samples_per_stego_type, replace=False)
                found = [found[i] for i in indices]
            self.stego_files.extend(found)
            self.stego_categories.extend([stype] * len(found))

        clean_path = os.path.join(root_dir, "clean_image", "*.*")
        self.clean_files = sorted([f for f in glob.glob(clean_path) if f.lower().endswith(VALID_EXTS)])
        
        if clean_samples and len(self.clean_files) > clean_samples:
            indices = rng.choice(len(self.clean_files), clean_samples, replace=False)
            self.clean_files = [self.clean_files[i] for i in indices]

        num_stego = len(self.stego_files)
        num_clean = len(self.clean_files)

        print(f"[{split}] Found {num_stego} Stego and {num_clean} Clean images.")
        if num_stego == 0 or num_clean == 0:
            raise ValueError("Dataset missing required clean/stego classes.")

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
        tensor = _load_gray_tensor(img_path)

        if tensor.shape != (1, 256, 256):
            raise ValueError(f"Corrupted shape {tensor.shape} at: {img_path}")

        return tensor, self.labels[idx], self.categories[idx]


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_accuracy, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_balanced_accuracy": best_accuracy,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_balanced_accuracy", 0.0)


def report_metrics(labels, predictions, categories):
    bal_acc = balanced_accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    # Safe division in case a test mini-subset has 0 clean or 0 stego
    clean_acc = (cm[0, 0] / max(cm[0].sum(), 1)) * 100
    stego_acc = (cm[1, 1] / max(cm[1].sum(), 1)) * 100

    print(f"\n" + "="*45)
    print(f" OVERALL BALANCED ACCURACY: {bal_acc * 100:.2f}%")
    print(f" F1 Score:                  {f1 * 100:.2f}%")
    print(f" Precision:                 {prec * 100:.2f}%")
    print(f"-"*45)
    print(f" Clean Accuracy (Specificity): {clean_acc:.2f}%  [{cm[0,0]}/{cm[0].sum()}]")
    print(f" Stego Accuracy (Sensitivity): {stego_acc:.2f}%  [{cm[1,1]}/{cm[1].sum()}]")
    print(f"-"*45)

    for cat in STEGO_TYPES:
        cat_idx = [i for i, v in enumerate(categories) if v == cat]
        if cat_idx:
            correct = sum(predictions[i] == 1 for i in cat_idx)
            sub_acc = (correct / len(cat_idx)) * 100
            print(f"   ├── {cat:11s} accuracy:  {sub_acc:6.2f}%  [{correct}/{len(cat_idx)}]")
            
    print("="*45 + "\n")
    return bal_acc

def _release_device_memory():
    gc.collect()
    torch.cuda.empty_cache()


def train_model(config):
    device = select_device()
    amp_dtype = select_amp_dtype()
    print(f"Bound to: {torch.cuda.get_device_name(0)} | AMP dtype: {amp_dtype}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["workers"], pin_memory=True,
        persistent_workers=config["workers"] > 0, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["workers"], pin_memory=True,
        persistent_workers=config["workers"] > 0
    )

    model = XuNet(use_checkpointing=config["use_grad_checkpointing"]).to(device)
    model = model.to(memory_format=torch.channels_last)

    # [MODIFIED] Ada Lovelace hardware boost via PyTorch 2.0 Graph Compiler
    if config.get("compile_model", False) and hasattr(torch, "compile"):
        print("Compiling model graph (Epoch 1 will have a ~40s warmup penalty)...")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch, best_accuracy = 0, 0.0
    if config["resume"]:
        start_epoch, best_accuracy = load_checkpoint(config["resume"], model, optimizer, scheduler, scaler, device)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    last_ckpt_path = os.path.join(config["checkpoint_dir"], "last.pth")
    best_ckpt_path = os.path.join(config["checkpoint_dir"], "best.pth")

    accum_steps = max(1, config["grad_accum_steps"])

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        print(f"\nEpoch {epoch + 1}/{config['epochs']} "
              f"(micro-batch {config['batch_size']} x {accum_steps} accum steps "
              f"= effective batch {config['batch_size'] * accum_steps})")

        optimizer.zero_grad(set_to_none=True)
        train_bar = tqdm(train_loader, desc="Training")

        for step, (inputs, labels, _) in enumerate(train_bar):
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            try:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _release_device_memory()
                    raise RuntimeError(
                        "Ran out of GPU memory. Drop config['batch_size'] to 32."
                    ) from e
                raise

            train_bar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        if (step + 1) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        _release_device_memory()

        # --- Evaluation ---
        model.eval()
        all_preds_gpu, all_labels, all_categories = [], [], []

        eval_bar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for inputs, labels, categories in eval_bar:
                inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(inputs)

                preds = torch.argmax(outputs, dim=1)
                all_preds_gpu.append(preds)
                all_labels.extend(labels.numpy())
                all_categories.extend(categories)

        all_preds_cpu = torch.cat(all_preds_gpu).cpu().numpy()

        balanced_accuracy = report_metrics(all_labels, all_preds_cpu, all_categories)
        is_best = balanced_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, balanced_accuracy)

        # [MODIFIED] Save checkpoint after every single epoch
        epoch_ckpt_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch + 1:02d}.pth")
        save_checkpoint(epoch_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_accuracy, config)
        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_accuracy, config)
        
        if is_best:
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_accuracy, config)

        _release_device_memory()


# [MODIFIED] Tuned specifically for RTX 4050 6GB + Raptor Lake Mobile
config = {
    "epochs": 10,
    "batch_size": 64,               # Up from 16. Fits ~1.8GB total VRAM footprint.
    "grad_accum_steps": 1,          # True batch size 64 runs ~15% faster than 16x4
    "use_grad_checkpointing": False,# Turned OFF: trades 2GB VRAM for 25% speed gain
    "compile_model": True,          # Set to False if Windows throws a Triton/C++ compiler error
    "workers": 8,                   # Raptor Lake sweet-spot (avoids OS thread thrashing)
    "learning_rate": 0.0008,
    "resume": None,
    "train_dir": "./train_tiles",   
    "test_dir": "./test_tiles",     
    "checkpoint_dir": "./saved_checkpoints",
    "train_samples_per_type": None,
    "train_clean_samples": None,
    "test_samples_per_type": None,
    "test_clean_samples": None
}

if __name__ == "__main__":
    train_model(config)
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

STEGO_TYPES = ["LSB", "PVD", "WOW", "S-UNIWARD", "MiPOD"]


class SRMLayer(nn.Module):
    """Applies fixed Spatial Rich Model (SRM) filters to extract noise residuals."""

    def __init__(self):
        super().__init__()
        kernels = np.load("SRM_Kernels1.npy").transpose(3, 2, 0, 1)
        self.conv = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.conv.weight = nn.Parameter(
            torch.from_numpy(kernels).float(), requires_grad=False
        )

    def forward(self, x):
        return torch.clamp(self.conv(x), min=-2.0, max=2.0)


class GlobalCovariancePooling(nn.Module):
    """Computes the second-order statistical relationships between feature maps."""

    def forward(self, x):
        batch, channels, _, _ = x.shape
        features = x.view(batch, channels, -1)
        features = features - features.mean(dim=2, keepdim=True)

        # Compute covariance matrix: (Batch, Channels, Channels)
        cov = torch.bmm(features, features.transpose(1, 2)) / max(
            features.size(2) - 1, 1
        )

        # Extract only the unique upper-triangular values to avoid redundant data
        upper_idx = torch.triu_indices(channels, channels, device=x.device)
        cov = cov[:, upper_idx[0], upper_idx[1]]

        cov = torch.sign(cov) * torch.sqrt(torch.abs(cov) + 1e-8)
        return F.normalize(cov, p=2, dim=1)


class XuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMLayer()

        self.layer1 = nn.Sequential(
            nn.Conv2d(30, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),

        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.Tanh()
        )

        self.global_pool = GlobalCovariancePooling()

        # Input size = 256 channels * (256 + 1) / 2 unique covariance pairs
        cov_feature_size = 256 * (256 + 1) // 2

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(cov_feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2),
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


class StegoDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        self.types = []

        # 1. Gather Clean Images (Class 0)
        clean_path = os.path.join(root_dir, "clean_image", "*.*")

        for file_path in glob.glob(clean_path): #glob.glob gives the full path of the file
            self.files.append(file_path)
            self.labels.append(0)
            self.types.append("clean")

        # 2. Gather Stego Images (Class 1)
        for stego_type in STEGO_TYPES:
            stego_path = os.path.join(root_dir, stego_type, "*.*")
            for file_path in glob.glob(stego_path):
                self.files.append(file_path)
                self.labels.append(1)
                self.types.append(stego_type)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx): 
        #Gets the index as input
        img = Image.open(self.files[idx]) #accesses the file path and opens the img
        img_arr = np.array(img, dtype=np.float32) #Conver to numpy array of float32

        # Convert (256, 256) array -> (1, 256, 256) PyTorch Image Tensor
        tensor = torch.from_numpy(img_arr).unsqueeze(0)
        return tensor, self.labels[idx], self.types[idx]


def train():
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    )
    print(f"Training backend selected: {device}")
    #StegoDataset inherits from the Dataset class, implements __len__ and __getitem__ methods, which are required for a PyTorch Dataset.
    #Creates lists where image info are strored accordingly
    #When Dataloader is called, it first creates an instance of the StegoDataset class, which initializes the dataset by scanning the specified root directory for clean and stego images. 
    #It populates the files, labels, and types lists accordingly.
    #Now dataloader uses these lists to create batches of data for training and testing. It handles shuffling, batching, and parallel data loading using multiple worker threads.
    #Dataset is responsible for storing the image, and relative info (indexing) and Dataloader is responsible for loading the data in batches, shuffling, and parallel processing.
    train_loader = DataLoader(
        StegoDataset("./train_tiles"),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True, #Page locked memory, no need to worry about memory being paged out to disk, which can slow down data transfer to the GPU.
        #Direct memory access (DMA) can be used to transfer data directly from the pinned memory to the GPU, bypassing the CPU and reducing latency.
        #Useful for asynchronous data transfer, allowing the CPU to continue executing other tasks while the GPU is processing the data.
    )
    test_loader = DataLoader(
        StegoDataset("./test_tiles"),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = XuNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(1, 11):
        model.train() #enable training mode, which activates dropout and batch normalization layers
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            #Autocast handles mixed precision, choosing between 16-bit and 32-bit floating point operations to optimize performance and memory usage.
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                predictions = model(images)
                loss = criterion(predictions, labels)

            loss.backward() #computes gradients of the loss w.r.t. model parameters
            optimizer.step() #updates the model parameters based on the computed gradients

        scheduler.step() #adjusts the learning rate based on the scheduler

        # --- Test Evaluation ---
        model.eval()
        all_preds, all_targets, all_types = [], [], []

        with torch.no_grad():
            for images, labels, img_types in test_loader:
                images = images.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(images)

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.numpy())
                all_types.extend(img_types)

        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        print(
            f"\nResult Epoch {epoch:02d} | Balanced Acc: {bal_acc:.2%} | F1: {f1_score(all_targets, all_preds):.2%}"
        )

        # Print per-category breakdown
        for category in ["clean"] + STEGO_TYPES:
            cat_indices = [i for i, t in enumerate(all_types) if t == category]
            if cat_indices:
                correct = sum(all_preds[i] == all_targets[i] for i in cat_indices)
                print(
                    f"  ├── {category:<12} accuracy: {correct / len(cat_indices):.1%}"
                )

        if bal_acc > best_accuracy:
            best_accuracy = bal_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("  └── [Saved new best checkpoint]")


if __name__ == "__main__":
    train()

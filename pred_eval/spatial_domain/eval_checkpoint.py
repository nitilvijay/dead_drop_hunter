import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from nvidia_gpu_train import XuNet
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

checkpoint_folder = os.path.join("/home/nitil/brainfuel/dead_drop_hunter/", "saved_checkpoints_neo_2")
SUB_FOLDER_NAMES = ["LSB", "PVD", "WOW", "S-UNIWARD", "MiPOD", "clean_image"]

#select device
def select_pred_device():
    
    if torch.xpu.is_available():
        print("[Info] Intel XPU is available. Using XPU for predictions.")
        device = torch.device("xpu")
        return device
    else:
        print("[Info] Intel XPU is not available. Using CPU for predictions.")
        device = torch.device("cpu")
        return device

def load_model(checkpoint_path, device):
    checkpoint_details = torch.load(checkpoint_path, map_location=device)
    
    '''What is map_location, since the model was trained using a nvidia based
    gpu while saving the tensors, pytroch remebers it was saved with CUDA, because
    to_device() pointed to cuda, now when trying to load on a new machine, the pytorch
    tries to load on cuda and searches for it, but not found.'''
    
    model = XuNet().to(device)
    model.load_state_dict(checkpoint_details["model_state_dict"])
    model.eval()
    return model, checkpoint_details["epoch"]

def load_image(image_path):
    img = Image.open(image_path)
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr)
    
    tensor = tensor.unsqueeze(0).unsqueeze(0) #Add channel and batch value as 1
    return tensor

def report_metrics(labels, predictions, categories):
    bal_acc = balanced_accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    clean_acc = (cm[0, 0] / max(cm[0].sum(), 1)) * 100
    stego_acc = (cm[1, 1] / max(cm[1].sum(), 1)) * 100

    print(f"\n" + "=" * 45)
    print(f" OVERALL BALANCED ACCURACY: {bal_acc * 100:.2f}%")
    print(f" F1 Score:                  {f1 * 100:.2f}%")
    print(f" Precision:                 {prec * 100:.2f}%")
    print(f"-" * 45)
    print(f" Clean Accuracy (Specificity): {clean_acc:.2f}%  [{cm[0,0]}/{cm[0].sum()}]")
    print(f" Stego Accuracy (Sensitivity): {stego_acc:.2f}%  [{cm[1,1]}/{cm[1].sum()}]")
    print(f"-" * 45)

    for cat in SUB_FOLDER_NAMES[:-1]:
        cat_idx = [i for i, v in enumerate(categories) if v == cat]
        if cat_idx:
            correct = sum(predictions[i] == 1 for i in cat_idx)
            sub_acc = (correct / len(cat_idx)) * 100
            print(f"   ├── {cat:11s} accuracy:  {sub_acc:6.2f}%  [{correct}/{len(cat_idx)}]")

    print("=" * 45 + "\n")

device  = select_pred_device()
for _, _, files in os.walk(checkpoint_folder):
    for file in files:
        model, epoch_file_name = load_model(os.path.join(checkpoint_folder, file), device)
        print(f"Loaded model from: {epoch_file_name}")
        print(f"Checkpoint file: {file}")
        #Load images in batches from the test folder
        test_folder_path = os.path.join("/home/nitil/brainfuel/dead_drop_hunter/", "test_tiles_1")
        
        #For report metric function, we need to keep track of the labels, predictions and categories
        labels = []
        predictions = []
        categories = []
        
        for image_sub_folder in SUB_FOLDER_NAMES:
    
            is_clean_folder = image_sub_folder == "clean_image"
            folder_label = 0 if is_clean_folder else 1
            
            image_sub_folder_path = os.path.join(test_folder_path, image_sub_folder)

            for _, _, image_files in os.walk(image_sub_folder_path):
                batch_tensors = []
                count = 0
                for image_file in tqdm(image_files, desc=f"Processing {image_sub_folder}"):
                    image_path = os.path.join(image_sub_folder_path, image_file)
                    tensor = load_image(image_path)
                    batch_tensors.append(tensor)
                    
                    count += 1
                    if count == 24:  # Process in batches of 24
                        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
                        #moved to batch to device before prediction
                        
                        with torch.no_grad():
                            model_outputs = model(batch_tensor)
                            probabilities = torch.softmax(model_outputs, dim=1)

                        batch_size = probabilities.size(0)
                        labels.extend([folder_label] * batch_size)
                        categories.extend([image_sub_folder] * batch_size)
                        predictions.extend((probabilities[:, 1] > 0.5).to(torch.int64).cpu().tolist())
                                    
                        batch_tensors = []
                        count = 0
                
                # Handle last incomplete batch
                if batch_tensors:
                    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
                    
                    with torch.no_grad():
                        model_outputs = model(batch_tensor)
                        probabilities = torch.softmax(model_outputs, dim=1)

                    batch_size = probabilities.size(0)
                    labels.extend([folder_label] * batch_size)
                    categories.extend([image_sub_folder] * batch_size)
                    predictions.extend((probabilities[:, 1] > 0.5).to(torch.int64).cpu().tolist())

        report_metrics(labels, predictions, categories)

#load_model("/home/nitil/brainfuel/dead_drop_hunter/saved_checkpoints_neo_1/epoch_07.pth", select_pred_device())
# image_path = input("Enter the image Path: ").strip("'")
# print(image_path)
# load_image(image_path)
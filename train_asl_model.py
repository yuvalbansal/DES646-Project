import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==========================
# CONFIGURATION
# ==========================
DATA_DIR = "dataset_trimmed"                   # path to dataset (train/, test/)
SAVE_DIR = "runs/checkpoints"
MODEL_NAME = "mobilenetv3_small"
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================
# DATA PIPELINE
# ==========================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

num_classes = len(train_dataset.classes)
print(f"Found {num_classes} classes: {train_dataset.classes}")


# ==========================
# MODEL SETUP
# ==========================
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                 patience=2, factor=0.5)


# ==========================
# TRAINING LOOP
# ==========================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step(test_acc)

    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f} | "
          f"Time: {(time.time()-t0):.1f}s")

    # Save best checkpoint
    if test_acc > best_acc:
        best_acc = test_acc
        ckpt_path = Path(SAVE_DIR) / "best.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": train_dataset.classes,
            "img_size": IMG_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "test_acc": best_acc,
            "epoch": epoch + 1,
            "model_name": MODEL_NAME,
        }, ckpt_path)
        print(f"✅ Saved best model to {ckpt_path} (Test Acc: {best_acc:.4f})")


# ==========================
# EXPORT LABELS + METADATA
# ==========================
metadata = {
    "num_classes": num_classes,
    "img_size": IMG_SIZE,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "class_names": train_dataset.classes,
    "best_acc": best_acc,
    "epochs": NUM_EPOCHS,
}
with open(Path(SAVE_DIR) / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("🎉 Training complete. Model and metadata saved in", SAVE_DIR)

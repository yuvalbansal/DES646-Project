import os, random, shutil
from pathlib import Path

DATA_DIR = Path("dataset/asl_alphabets")  # Original Kaggle dataset
OUT_DIR = Path("dataset")
TRAIN_RATIO = 0.8  # 80% training, 20% validation

for class_dir in DATA_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    images = list(class_dir.glob("*.jpg"))
    random.shuffle(images)
    split = int(len(images) * TRAIN_RATIO)
    
    train_imgs = images[:split]
    val_imgs = images[split:]
    
    for subset, imgs in [("train", train_imgs), ("test", val_imgs)]:
        dest = OUT_DIR / subset / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest / img.name)

print("✅ Dataset split complete!")

"""
Merge 'Pothole Dataset' (folder with images + .txt labels) into data/pothole_dataset.
Splits into train/val (80/20), copies images and labels to the right places.
Run this, then run prepare_combined_dataset.py and train.
"""

import random
import shutil
from pathlib import Path

POTHOLE_DATASET = Path("data/pothole_dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def merge_pothole_dataset(
    source_dir: str = "Pothole Dataset",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    # Find all image files that have a matching .txt label
    pairs = []
    for f in source.iterdir():
        if not f.is_file() or f.suffix.lower() not in EXTENSIONS:
            continue
        lbl = source / (f.stem + ".txt")
        if lbl.exists():
            pairs.append((f, lbl))
    if not pairs:
        raise FileNotFoundError(f"No image+label pairs found in {source}")

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, pair_list in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = POTHOLE_DATASET / "images" / split
        lbl_dir = POTHOLE_DATASET / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pair_list:
            dest_img = img_dir / img_path.name
            dest_lbl = lbl_dir / (img_path.stem + ".txt")
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)

    print(f"Merged '{source_dir}' into {POTHOLE_DATASET}")
    print(f"  Train: {len(train_pairs)} images + labels -> images/train, labels/train")
    print(f"  Val:   {len(val_pairs)} images + labels -> images/val, labels/val")
    print("\nNext steps:")
    print("  1. python prepare_combined_dataset.py")
    print("  2. python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Merge Pothole Dataset folder into data/pothole_dataset")
    p.add_argument("--source", default="Pothole Dataset", help="Folder containing images and .txt labels")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    merge_pothole_dataset(source_dir=args.source, val_ratio=args.val_ratio, seed=args.seed)

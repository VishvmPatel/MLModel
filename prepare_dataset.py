"""
Prepare pothole dataset for YOLOv8.
- Splits images into train/val (default 80/20).
- Creates YOLO directory structure: images/train, images/val, labels/train, labels/val.
- Copies images and creates empty label files (same base name) for annotation.
Run this first, then annotate with LabelImg/Roboflow (YOLO format).
"""

import os
import shutil
import random
from pathlib import Path


def prepare_dataset(
    images_dir: str = "Images",
    output_dir: str = "data/pothole_dataset",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    images_path = Path(images_dir)
    out = Path(output_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    # Supported image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in exts]
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}")

    random.seed(seed)
    random.shuffle(image_files)
    n_val = max(1, int(len(image_files) * val_ratio))
    val_files = set(image_files[:n_val])
    train_files = [f for f in image_files if f not in val_files]

    for split, files in [("train", train_files), ("val", val_files)]:
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest_img = img_dir / f.name
            if not dest_img.exists() or dest_img.stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, dest_img)
            # Create empty label file if not present (for annotation)
            lbl_file = lbl_dir / (f.stem + ".txt")
            if not lbl_file.exists():
                lbl_file.touch()

    # Update dataset yaml path to absolute
    yaml_path = Path("data/pothole.yaml")
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            content = f.read()
        abs_path = str(out.resolve()).replace("\\", "/")
        if "path:" in content and "path: ." in content:
            content = content.replace("path: .", f"path: {abs_path}", 1)
            with open(yaml_path, "w") as f:
                f.write(content)

    print(f"Dataset prepared: {output_dir}")
    print(f"  Train: {len(train_files)} images -> {out / 'images' / 'train'}")
    print(f"  Val:   {len(val_files)} images -> {out / 'images' / 'val'}")
    print(f"  Labels: {out / 'labels' / 'train'} and {out / 'labels' / 'val'}")
    print("  Add YOLO-format .txt annotations (class_id x_center y_center width height, normalized 0-1).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepare pothole dataset for YOLOv8")
    p.add_argument("--images-dir", default="Images", help="Folder containing images")
    p.add_argument("--output-dir", default="data/pothole_dataset", help="Output dataset root")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    prepare_dataset(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

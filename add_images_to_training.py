"""
Add one or more images to the pothole training set using the current model's predictions as labels.
Use this to add your own test images (e.g. close-up potholes) so the model gets better and more confident.
After running, re-run prepare_combined_dataset.py and then train to improve the model.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

POTHOLE_DATASET = Path("data/pothole_dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def add_images_to_training(
    image_paths: list[Path],
    weights: str = "runs/detect/combined/weights/best.pt",
    conf: float = 0.15,
    device: str = "",
    prefix: str = "added",
) -> int:
    """Run model on each image, save predicted boxes as YOLO labels, copy image to pothole_dataset/images/train."""
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = YOLO(str(weights_path))

    train_img = POTHOLE_DATASET / "images" / "train"
    train_lbl = POTHOLE_DATASET / "labels" / "train"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    added = 0
    for i, img_path in enumerate(image_paths):
        if not img_path.exists():
            print(f"  Skip (not found): {img_path}")
            continue
        if img_path.suffix.lower() not in EXTENSIONS:
            print(f"  Skip (not image): {img_path}")
            continue

        results = model.predict(source=str(img_path), conf=conf, device=device, verbose=False)
        lines = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                xywhn = r.boxes.xywhn.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for j in range(len(cls)):
                    c = int(cls[j])
                    xc, yc, w, h = xywhn[j]
                    lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            print(f"  Skip (no detections): {img_path}")
            continue

        stem = f"{prefix}_{img_path.stem}_{i}"
        dest_name = stem + img_path.suffix
        dest_img = train_img / dest_name
        dest_lbl = train_lbl / (stem + ".txt")
        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Added: {img_path.name} -> {dest_name} ({len(lines)} box(es))")
        added += 1

    return added


def main():
    parser = argparse.ArgumentParser(
        description="Add images to pothole training set using current model's predictions as labels"
    )
    parser.add_argument("images", nargs="+", help="Image file(s) to add (e.g. your pothole photo)")
    parser.add_argument("--weights", default="runs/detect/combined/weights/best.pt", help="Model weights")
    parser.add_argument("--conf", type=float, default=0.15, help="Min confidence to keep a detection (default 0.15)")
    parser.add_argument("--device", default="")
    parser.add_argument("--prefix", default="added", help="Filename prefix for added images")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    # Expand folders
    expanded = []
    for p in paths:
        if p.is_dir():
            expanded.extend([f for f in p.iterdir() if f.suffix.lower() in EXTENSIONS])
        else:
            expanded.append(p)
    if not expanded:
        print("No images to add.")
        return
    print(f"Adding {len(expanded)} image(s) to pothole training set...")
    n = add_images_to_training(
        expanded,
        weights=args.weights,
        conf=args.conf,
        device=args.device,
        prefix=args.prefix,
    )
    print(f"\nAdded {n} image(s) to {POTHOLE_DATASET / 'images' / 'train'}.")
    print("Next steps to improve the model:")
    print("  1. python prepare_combined_dataset.py")
    print("  2. python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device 0")
    print("  (Optionally refine labels in data/pothole_dataset/labels/train/ in LabelImg before step 1.)")


if __name__ == "__main__":
    main()

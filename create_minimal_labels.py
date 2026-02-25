"""
Create minimal YOLO-format labels so training can run when no annotations exist.
Adds one placeholder bounding box per image (center, 20% size) for a subset of train/val.
Only touches empty .txt files. Replace with real labels for proper training.
"""

from pathlib import Path

DATASET = Path("data/pothole_dataset")
# YOLO format: class_id x_center y_center width height (normalized 0-1)
# One small box at center so the image is not "empty" for the trainer
PLACEHOLDER_LINE = "0 0.5 0.5 0.2 0.2\n"


def main():
    for split in ["train", "val"]:
        lbl_dir = DATASET / "labels" / split
        img_dir = DATASET / "images" / split
        if not lbl_dir.exists():
            continue
        count = 0
        for txt in lbl_dir.glob("*.txt"):
            if txt.stat().st_size == 0:
                txt.write_text(PLACEHOLDER_LINE)
                count += 1
                # Limit to 500 train / 100 val so training runs but we don't overwrite too many
                if split == "train" and count >= 500:
                    break
                if split == "val" and count >= 100:
                    break
        print(f"  {split}: wrote placeholder labels to {count} files.")
    print("Run: python train.py (use real labels for production accuracy).")


if __name__ == "__main__":
    main()

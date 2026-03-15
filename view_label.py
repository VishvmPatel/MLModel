"""
Visualize the ground-truth YOLO label boxes for a single image.

Usage:
    python view_label.py data/combined_dataset/images/train/electricpole_000000.jpg

This will:
- Find the matching label file under data/combined_dataset/labels/{train,val}/
- Draw the labeled boxes on the image with class names
- Save the result under results/label_viz/ with the same filename
"""

import sys
from pathlib import Path

import cv2


CLASS_NAMES = {
    0: "pothole",
    1: "fallen_tree",
    2: "electric_pole",
}


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Convert normalized YOLO (xc, yc, w, h) to pixel (x1, y1, x2, y2)."""
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)
    return x1, y1, x2, y2


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_label.py path/to/image.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    # Infer label path assuming combined_dataset layout
    # images/train/electricpole_000000.jpg -> labels/train/electricpole_000000.txt
    try:
        parts = list(img_path.parts)
        # Replace "images" with "labels"
        for i, p in enumerate(parts):
            if p == "images":
                parts[i] = "labels"
                break
        lbl_path = Path(*parts).with_suffix(".txt")
    except Exception:
        lbl_path = img_path.with_suffix(".txt")

    if not lbl_path.exists():
        print(f"Label file not found for image: {lbl_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        color = (0, 255, 0)  # green box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    out_dir = Path("results/label_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"Labeled image saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()


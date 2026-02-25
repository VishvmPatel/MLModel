"""
Auto-annotate images using a pothole detection model: run the model on each image
and save predicted boxes as YOLO-format label files. Optionally expand boxes so they
cover more of each pothole (full extent). You can then refine these in LabelImg.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def expand_box(xc: float, yc: float, w: float, h: float, margin: float):
    """Expand box by margin (e.g. 0.15 = 15% on each side). Clamp to [0,1]."""
    w2 = min(1.0, w * (1 + 2 * margin))
    h2 = min(1.0, h * (1 + 2 * margin))
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    x1 = max(0.0, xc - w2 / 2)
    y1 = max(0.0, yc - h2 / 2)
    x2 = min(1.0, xc + w2 / 2)
    y2 = min(1.0, yc + h2 / 2)
    new_xc = (x1 + x2) / 2
    new_yc = (y1 + y2) / 2
    new_w = x2 - x1
    new_h = y2 - y1
    return new_xc, new_yc, new_w, new_h


def auto_annotate(
    weights: str = "runs/detect/pothole/weights/best.pt",
    dataset_root: str = "data/pothole_dataset",
    conf: float = 0.2,
    expand_margin: float = 0.2,
    imgsz: int = 640,
    device: str = "",
) -> None:
    """
    Run model on all images in dataset_root/images/train and images/val,
    write YOLO-format .txt labels to dataset_root/labels/train and labels/val.
    expand_margin: expand each box by this fraction (0.2 = 20% each side) to cover more of pothole.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    path_weights = Path(weights)
    if not path_weights.is_absolute():
        path_weights = Path.cwd() / path_weights
    if not path_weights.exists():
        alt = Path.home() / "runs" / "detect" / "runs" / "detect" / "pothole" / "weights" / "best.pt"
        if alt.exists():
            path_weights = alt
        else:
            raise FileNotFoundError(f"Weights not found: {weights}. Train a model first.")
    model = YOLO(str(path_weights))

    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in exts]
        for i, img_path in enumerate(images):
            results = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )
            lines = []
            for r in results:
                if r.boxes is None:
                    continue
                xywhn = r.boxes.xywhn.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for j in range(len(cls)):
                    c = int(cls[j])
                    xc, yc, w, h = xywhn[j]
                    if expand_margin > 0:
                        xc, yc, w, h = expand_box(xc, yc, w, h, expand_margin)
                    lines.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  {split}: {i + 1}/{len(images)}")
    print("Done. Labels written to", root / "labels")
    print("Tip: Open a few images in LabelImg and correct any boxes that are still wrong or incomplete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-annotate images using model predictions (YOLO format)")
    parser.add_argument("--weights", default="runs/detect/pothole/weights/best.pt", help="Model weights")
    parser.add_argument("--dataset", default="data/pothole_dataset", help="Dataset root (images/train, images/val)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--expand", type=float, default=0.2,
                        help="Expand each box by this fraction (0.2 = 20% each side) to cover full pothole")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    auto_annotate(
        weights=args.weights,
        dataset_root=args.dataset,
        conf=args.conf,
        expand_margin=args.expand,
        imgsz=args.imgsz,
        device=args.device,
    )

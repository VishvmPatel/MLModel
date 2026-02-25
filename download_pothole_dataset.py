"""
Download a public pothole dataset (YOLO format) and merge into data/pothole_dataset.
Uses jaygala24/pothole-detection release (direct ZIP, no API key).
"""

import zipfile
import shutil
from pathlib import Path

try:
    import requests
except ImportError:
    from urllib.request import urlretrieve
    requests = None

DATASET_URL = "https://github.com/jaygala24/pothole-detection/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip"
OUTPUT_DIR = Path("data/pothole_dataset")
ZIP_PATH = Path("data/pothole_dataset_download.zip")


def download_and_merge():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading pothole dataset (YOLO format)...")
    if requests:
        r = requests.get(DATASET_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        from urllib.request import urlretrieve
        urlretrieve(DATASET_URL, ZIP_PATH)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = z.namelist()
        # Common layouts: Dataset/train/images, Dataset/train/labels, or images/, labels/
        for name in names:
            if name.endswith("/"):
                continue
            base = name.split("/")[0] if "/" in name else "."
            break
        z.extractall("data/")
    ZIP_PATH.unlink(missing_ok=True)
    extracted = Path("data")
    # Find extracted folder (often Pothole.Dataset.IVCNZ or similar)
    dirs = [d for d in extracted.iterdir() if d.is_dir() and d.name != "pothole_dataset"]
    if not dirs:
        # Maybe extracted into data/ with train/val already
        for d in extracted.iterdir():
            if d.is_dir() and (d / "images").exists() or (d / "train").exists():
                merge_from = d
                break
        else:
            merge_from = None
    else:
        merge_from = max(dirs, key=lambda d: len(list(d.rglob("*.*"))))
    if merge_from is None:
        # List and use first dir that has images
        for d in extracted.iterdir():
            if d.is_dir():
                imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
                if imgs:
                    merge_from = d
                    break
        else:
            merge_from = dirs[0] if dirs else None
    if merge_from is None:
        print("Could not find extracted images/labels. Check data/ manually.")
        return
    # Merge into YOLO structure: images/train, images/val, labels/train, labels/val
    (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)
    train_img = OUTPUT_DIR / "images" / "train"
    train_lbl = OUTPUT_DIR / "labels" / "train"
    val_img = OUTPUT_DIR / "images" / "val"
    val_lbl = OUTPUT_DIR / "labels" / "val"
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    for ext in exts:
        all_images.extend(merge_from.rglob(f"*{ext}"))
    if not all_images:
        # Maybe structure is train/images, train/labels
        for sub in ["train", "Train"]:
            t = merge_from / sub
            if t.exists():
                for ext in exts:
                    all_images.extend((t / "images").rglob(f"*{ext}") if (t / "images").exists() else t.rglob(f"*{ext}"))
        if not all_images:
            all_images = list(merge_from.rglob("*.jpg")) + list(merge_from.rglob("*.png"))
    n = len(all_images)
    n_val = max(1, n // 5)
    for i, img_path in enumerate(all_images):
        stem = img_path.stem
        lbl_path = img_path.parent / (stem + ".txt")
        if not lbl_path.exists():
            # Try labels folder
            for labels_dir in [img_path.parent.parent / "labels", img_path.parent.parent / "Labels", merge_from / "labels"]:
                if (labels_dir / (stem + ".txt")).exists():
                    lbl_path = labels_dir / (stem + ".txt")
                    break
        if i < n - n_val:
            dest_img = train_img / img_path.name
            dest_lbl = train_lbl / (stem + ".txt")
        else:
            dest_img = val_img / img_path.name
            dest_lbl = val_lbl / (stem + ".txt")
        if not dest_img.exists() or img_path.stat().st_mtime > dest_img.stat().st_mtime:
            shutil.copy2(img_path, dest_img)
        if lbl_path.exists() and (not dest_lbl.exists() or lbl_path.stat().st_mtime > dest_lbl.stat().st_mtime):
            shutil.copy2(lbl_path, dest_lbl)
    # Cleanup extracted folder
    try:
        shutil.rmtree(merge_from)
    except Exception:
        pass
    print(f"Merged {n} images into {OUTPUT_DIR} (train: {n - n_val}, val: {n_val}).")
    print("You can now run: python train.py")


if __name__ == "__main__":
    download_and_merge()

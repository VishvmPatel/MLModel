"""
Run LabelImg with combined_dataset folders and class list.
Usage: python run_labelImg.py

LabelImg expects: [image_dir] [class_file] [save_dir]
We pass: images/train, classes.txt, labels/train so boxes save to the right place.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "combined_dataset" / "images" / "train"
LABELS = ROOT / "data" / "combined_dataset" / "labels" / "train"
CLASSES = ROOT / "data" / "combined_dataset" / "classes.txt"

def main():
    try:
        # LabelImg CLI: labelImg.py [image_dir] [class_file] [save_dir]
        if len(sys.argv) < 2:
            sys.argv[1:] = [
                str(IMAGES),
                str(CLASSES) if CLASSES.exists() else "",
                str(LABELS),
            ]
        import labelImg.labelImg as lbl
        lbl.main()
    except Exception as e:
        import traceback
        print("LabelImg crashed with:")
        traceback.print_exc()
        input("Press Enter to close...")
        sys.exit(1)

if __name__ == "__main__":
    main()

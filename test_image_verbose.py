"""
Run detection on ONE image with very low confidence (0.01) and print exactly what was detected and where the output is.
Use this when you see no boxes - it tells you if the model outputs anything at all.
"""

import sys
from pathlib import Path

from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image_verbose.py path/to/image.jpg")
        sys.exit(1)
    source = Path(sys.argv[1])
    if not source.exists():
        print(f"Error: File not found: {source}")
        sys.exit(1)

    weights = Path("runs/detect/combined/weights/best.pt")
    if not weights.exists():
        print(f"Error: Weights not found: {weights}")
        sys.exit(1)

    out_dir = Path("runs/detect/single_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / source.name

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=0.01,  # very low - show any detection
        device="0",
        save=False,
    )

    total_boxes = 0
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            n = len(r.boxes)
            total_boxes += n
            for i in range(n):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                names = r.names or {}
                label = names.get(cls_id, str(cls_id))
                print(f"  Detection: {label} (confidence {conf:.3f})")
        # Save image (with or without boxes)
        im = r.plot()
        if im is not None:
            try:
                import cv2
                cv2.imwrite(str(out_file), im)
            except Exception:
                from PIL import Image
                Image.fromarray(im).save(out_file)

    if total_boxes == 0:
        print("No detections (model did not detect pothole / fallen_tree / electric_pole on this image).")
        print("This often happens when the image looks different from training data (e.g. unpaved road vs asphalt).")
        print("Add this image to training and label the objects, then re-train.")
    else:
        print(f"Total: {total_boxes} detection(s).")

    print(f"\nOutput image saved to: {out_file.resolve()}")
    print("Open that file to see the result (boxes drawn if any were detected).")


if __name__ == "__main__":
    main()

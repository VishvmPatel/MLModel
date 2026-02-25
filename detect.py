"""
Run detection with YOLOv8 and outline (draw bounding boxes on) detections.
Supports single-class (e.g. pothole) or multi-class (e.g. pothole + fallen_tree) models.
Use the trained best.pt or any exported model.
For app accuracy: use higher --conf (e.g. 0.45) and optionally --max-area to drop oversized false positives.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def _filter_boxes_by_area(result, max_area_ratio: float):
    """Keep only boxes that cover at most max_area_ratio of the image (0-1). Drops huge false positives."""
    if result.boxes is None or len(result.boxes) == 0:
        return
    h, w = result.orig_shape[:2]
    img_area = h * w
    xyxy = result.boxes.xyxy.cpu().numpy()
    keep = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= max_area_ratio * img_area:
            keep.append(i)
    if len(keep) < len(result.boxes):
        result.boxes = result.boxes[keep]


def detect(
    weights: str = "runs/detect/pothole/weights/best.pt",
    source: str = "Images",
    output_dir: str = "runs/detect/pothole_predict",
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "",
    save: bool = True,
    save_txt: bool = False,
    save_crop: bool = False,
    show: bool = False,
    line_width: int = 2,
    show_conf: bool = True,
    max_area: float | None = None,
) -> None:
    """
    Run inference and draw bounding boxes on images.
    max_area: if set (e.g. 0.25), drop any box covering more than this fraction of the image (reduces big false positives).
    """
    path_weights = Path(weights)
    if not path_weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}. Train first with: python train.py"
        )

    model = YOLO(str(path_weights))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If filtering by area, we need to run without save then filter and save ourselves
    if max_area is not None and max_area > 0:
        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            save=False,
            line_width=line_width,
        )
        for r in results:
            _filter_boxes_by_area(r, max_area)
            if save and r.path is not None:
                out_file = out_path / Path(r.path).name
                im = r.plot()
                if im is not None:
                    try:
                        import cv2
                        cv2.imwrite(str(out_file), im)
                    except Exception:
                        from PIL import Image
                        Image.fromarray(im).save(out_file)
    else:
        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            save=save,
            project=out_path.parent,
            name=out_path.name,
            exist_ok=True,
            save_txt=save_txt,
            save_conf=save_txt and show_conf,
            save_crop=save_crop,
            show=show,
            line_width=line_width,
        )

    print(f"\nPredictions saved to: {out_path.resolve()}")
    print("Detections are outlined with bounding boxes and labels (class + confidence).")
    if max_area is not None:
        print(f"Filtered out boxes larger than {max_area*100:.0f}% of image area.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and outline objects (e.g. potholes, fallen trees) with YOLOv8")
    parser.add_argument("--weights", default="runs/detect/pothole/weights/best.pt",
                        help="Model weights (best.pt after training)")
    parser.add_argument("--source", default="Images",
                        help="Image/dir/video path")
    parser.add_argument("--output-dir", default="runs/detect/pothole_predict",
                        help="Output directory for visualized images")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (use 0.4-0.5 for fewer false positives in apps)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-area", type=float, default=None, metavar="FRAC",
                        help="Drop boxes larger than this fraction of image (e.g. 0.25). Reduces huge false potholes.")
    parser.add_argument("--device", default="", help="cuda or cpu")
    parser.add_argument("--no-save", action="store_true", help="Do not save output images")
    parser.add_argument("--save-txt", action="store_true", help="Save labels as .txt")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--no-conf", action="store_true", help="Hide confidence in saved labels")
    args = parser.parse_args()

    detect(
        weights=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
        save=not args.no_save,
        save_txt=args.save_txt,
        line_width=args.line_width,
        show_conf=not args.no_conf,
        max_area=args.max_area,
    )

"""
Train YOLOv8 model for pothole detection with accuracy-focused settings.
Model will learn to detect and localize potholes (bounding boxes).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str = "data/pothole.yaml",
    model_size: str = "l",
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/detect",
    name: str = "pothole",
    exist_ok: bool = True,
    pretrained: bool = True,
    patience: int = 50,
    save_period: int = 10,
    amp: bool = True,
    seed: int = 42,
    resume: bool = False,
) -> str:
    """
    Train YOLOv8 for pothole detection.
    Uses larger model and more epochs for better accuracy.
    """
    # Resolve paths: use absolute so runs and plots save in project directory
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = Path.cwd() / project_path

    # Resume: load last.pt from the run directory
    if resume:
        last_pt = project_path / name / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(f"Cannot resume: {last_pt} not found. Train from scratch first.")
        model = YOLO(str(last_pt))
        print(f"Resuming from {last_pt}")
    else:
        # YOLOv8n=nano, s=small, m=medium, l=large, x=extra-large (best accuracy)
        model_name = f"yolov8{model_size}.pt"
        model = YOLO(model_name)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_path),
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        patience=patience,
        save_period=save_period,
        amp=amp,
        seed=seed,
        resume=resume,
        # Augmentation (improves accuracy)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # NMS at inference (keep default; lower conf threshold in detect.py if needed)
        verbose=True,
        workers=0,  # avoid RAM spikes / NumPy memory errors during validation
    )

    best_weights = project_path / name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nBest model saved to: {best_weights}")
        return str(best_weights)
    return str(project_path / name / "weights" / "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 pothole detector")
    parser.add_argument("--data", default="data/pothole.yaml", help="Dataset YAML")
    parser.add_argument("--model", default="l", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n=nano, x=best accuracy")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 or 1280 for small potholes)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="cuda device (0, 1, ... or cpu)")
    parser.add_argument("--project", default="runs/detect", help="Save project dir")
    parser.add_argument("--name", default="pothole", help="Run name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in project/name/weights")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )

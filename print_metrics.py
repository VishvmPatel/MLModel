"""
Run validation and print all performance metrics for the combined model (pothole, fallen_tree, electric_pole).
Also saves metrics to a text file.
Usage: python print_metrics.py
"""

from pathlib import Path
from ultralytics import YOLO


def _to_scalar(x):
    """Convert Ultralytics metric values (which may be arrays) to a single float."""
    if x is None:
        return None
    # If it's a numpy/torch array or list, take the mean
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        try:
            # Prefer .mean() if available
            if hasattr(x, "mean"):
                return float(x.mean())
            # Fallback to first element
            return float(x[0])
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return None

def main():
    weights = Path("runs/detect/combined/weights/best.pt")
    data_yaml = Path("data/combined.yaml")
    if not weights.exists():
        print(f"Weights not found: {weights}")
        return
    if not data_yaml.exists():
        print(f"Data YAML not found: {data_yaml}")
        return

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), imgsz=640, batch=16, plots=False)

    lines = []
    lines.append("=" * 60)
    lines.append("COMBINED MODEL PERFORMANCE METRICS")
    lines.append("(pothole, fallen_tree, electric_pole)")
    lines.append("=" * 60)

    # Overall box metrics
    if hasattr(metrics, "box") and metrics.box is not None:
        box = metrics.box
        lines.append("\n--- Overall (all classes) ---")
        m50 = _to_scalar(getattr(box, "ap50", None)) or _to_scalar(getattr(box, "map50", None))
        m5095 = _to_scalar(getattr(box, "ap", None)) or _to_scalar(getattr(box, "map", None))
        p = _to_scalar(getattr(box, "p", None)) or _to_scalar(getattr(box, "mp", None))
        r = _to_scalar(getattr(box, "r", None)) or _to_scalar(getattr(box, "mr", None))
        if m50 is not None:
            lines.append(f"  mAP50:        {m50:.4f}")
        if m5095 is not None:
            lines.append(f"  mAP50-95:     {m5095:.4f}")
        if p is not None:
            lines.append(f"  Precision:    {p:.4f}")
        if r is not None:
            lines.append(f"  Recall:       {r:.4f}")
    if hasattr(metrics, "fitness") and metrics.fitness is not None:
        lines.append(f"  Fitness:      {metrics.fitness:.4f}")

    # Per-class metrics (if available)
    if hasattr(metrics, "results_dict") and metrics.results_dict:
        lines.append("\n--- Per-class metrics ---")
        for k, v in metrics.results_dict.items():
            if isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Class-wise P, R, mAP50, mAP50-95 from results_dict or box
    if hasattr(metrics, "box"):
        box = metrics.box
        if hasattr(box, "ap50") and hasattr(box.ap50, "__len__"):
            names = getattr(model, "names", {}) or {}
            ap50 = box.ap50 if hasattr(box.ap50, "__len__") else [box.ap50]
            for i, ap in enumerate(ap50):
                name = names.get(i, f"class_{i}")
                lines.append(f"  {name} mAP50: {float(ap):.4f}")
        if hasattr(box, "p") and hasattr(box.p, "__len__"):
            names = getattr(model, "names", {}) or {}
            for i, p in enumerate(box.p):
                name = names.get(i, f"class_{i}")
                lines.append(f"  {name} Precision: {float(p):.4f}")
        if hasattr(box, "r") and hasattr(box.r, "__len__"):
            names = getattr(model, "names", {}) or {}
            for i, r in enumerate(box.r):
                name = names.get(i, f"class_{i}")
                lines.append(f"  {name} Recall: {float(r):.4f}")

    lines.append("\n" + "=" * 60)

    text = "\n".join(lines)
    print(text)

    # Save metrics to central results folder
    out_file = Path("results/combined_metrics.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8")
    print(f"\nMetrics saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()

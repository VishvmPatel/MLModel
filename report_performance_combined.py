"""
Generate a full performance report for the COMBINED YOLOv8 model (pothole + fallen_tree + electric_pole):
- Runs validation to produce confusion matrix, F1/P/R/PR curves
- Builds an HTML report with all graphs and key metrics
- Saves everything under results/combined_report
"""

import argparse
import base64
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _to_scalar(x):
    """Convert Ultralytics metric values (which may be arrays) to a single float."""
    if x is None:
        return None
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        try:
            if hasattr(x, "mean"):
                return float(x.mean())
            return float(x[0])
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return None


def find_run_dir(project: Path, name: str) -> Path | None:
    """Find the latest run directory (project/name or project/name2, name3, ...)."""
    if not project.exists():
        return None
    base = project / name
    if base.exists():
        return base
    # Ultralytics sometimes creates name2, name3 if exist_ok
    for d in sorted(project.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and d.name.startswith(name):
            return d
    return None


def run_validation_and_plots(
    weights: Path,
    data_yaml: Path,
    output_dir: Path,
    imgsz: int = 640,
    batch: int = 16,
) -> dict:
    """Run validation; save plots to output_dir. Returns metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    # Validation with plots=True saves confusion_matrix.png, F1_curve.png, P_curve.png, R_curve.png, PR_curve.png
    # Use absolute project path so plots save in project directory
    save_dir = output_dir.resolve()
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    # Ultralytics saves to project/name; copy into report dir if saved elsewhere
    possible_save = save_dir.parent / save_dir.name
    if not (save_dir / "confusion_matrix.png").exists() and possible_save.exists():
        for f in possible_save.glob("*.png"):
            shutil.copy(f, save_dir / f.name)
    # Plots saved to output_dir (project/name)
    # Build metrics dict for report (DetMetrics.box: ap50, ap, p, r)
    if hasattr(metrics, "box") and metrics.box is not None:
        box = metrics.box
        m50 = _to_scalar(getattr(box, "ap50", None)) or _to_scalar(getattr(box, "map50", None))
        m5095 = _to_scalar(getattr(box, "ap", None)) or _to_scalar(getattr(box, "map", None))
        p = _to_scalar(getattr(box, "p", None)) or _to_scalar(getattr(box, "mp", None))
        r = _to_scalar(getattr(box, "r", None)) or _to_scalar(getattr(box, "mr", None))
        out = {
            "mAP50": m50,
            "mAP50-95": m5095,
            "precision": p,
            "recall": r,
            "fitness": _to_scalar(getattr(metrics, "fitness", None)),
        }
    else:
        out = {}
    return out


def image_to_data_uri(path: Path) -> str:
    """Embed image as data URI."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def build_html_report(
    report_dir: Path,
    metrics: dict,
    run_dir: Path | None,
    val_plots_dir: Path,
) -> None:
    """Write combined_report.html with all graphs and metrics."""
    report_dir.mkdir(parents=True, exist_ok=True)
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Combined Model Performance Report</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 2rem; background: #0b1020; color: #eee; }",
        "h1 { color: #ffb703; }",
        "h2 { color: #8ecae6; margin-top: 2rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }",
        "table { border-collapse: collapse; margin: 1rem 0; }",
        "th, td { border: 1px solid #444; padding: 0.5rem 1rem; text-align: left; }",
        "th { background: #1d3557; color: #ffb703; }",
        "tr:nth-child(even) { background: #1b263b; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 1.5rem; margin: 1rem 0; }",
        ".grid img { max-width: 100%; height: auto; border: 1px solid #333; border-radius: 8px; }",
        ".card { background: #1b263b; padding: 1rem; border-radius: 8px; }",
        ".metric { font-size: 1.5rem; font-weight: bold; color: #ffb703; }",
        "</style></head><body>",
        "<h1>Combined model (pothole + fallen_tree + electric_pole) – performance report</h1>",
    ]

    # Exact model accuracy (primary metric: mAP50-95) as percentage
    accuracy_pct = None
    if metrics and metrics.get("mAP50-95") is not None:
        accuracy_pct = round(float(metrics["mAP50-95"]) * 100, 2)
        html_parts.append(
            f"<p class='metric' style='font-size:2rem; margin:1.5rem 0;'>"
            f"<strong>Model accuracy (mAP50-95): {accuracy_pct}%</strong></p>"
        )
        (report_dir / "accuracy_percent.txt").write_text(f"{accuracy_pct}", encoding="utf-8")

    # Metrics summary table
    html_parts.append("<h2>Validation metrics</h2>")
    if metrics:
        html_parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        for k, v in metrics.items():
            if v is not None:
                if isinstance(v, float):
                    v = f"{v:.4f}"
                html_parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html_parts.append("</table>")
    else:
        html_parts.append("<p>Run validation to see metrics.</p>")

    # Validation plots (confusion matrix, F1, P, R, PR, results; Ultralytics may use Box* prefix)
    html_parts.append("<h2>Validation plots</h2><div class='grid'>")
    curve_names = [
        ("confusion_matrix.png", "Confusion matrix"),
        ("confusion_matrix_normalized.png", "Confusion matrix (normalized)"),
        ("BoxF1_curve.png", "F1 curve"),
        ("BoxP_curve.png", "Precision curve"),
        ("BoxR_curve.png", "Recall curve"),
        ("BoxPR_curve.png", "Precision-Recall curve"),
        ("results.png", "Training results"),
    ]
    for name, title in curve_names:
        path = val_plots_dir / name
        if not path.exists() and name.startswith("Box"):
            path = val_plots_dir / name.replace("Box", "", 1)
        if path.exists():
            dest = report_dir / path.name
            if path.resolve() != dest.resolve():
                shutil.copy(path, dest)
            uri = image_to_data_uri(path)
            if uri:
                html_parts.append(
                    f"<div class='card'><h3>{title}</h3><img src='{uri}' alt='{title}' style='max-width:100%'/></div>"
                )
    html_parts.append("</div>")
    html_parts.append("<p style='color:#8ecae6; margin-top:0.5rem;'>Best value on each curve is at its peak. Training curves below show the best epoch marked with a red dot and dashed line.</p>")

    # Training run artifacts (labels, batch samples, training curves)
    if run_dir and run_dir.exists():
        html_parts.append("<h2>Training run</h2><div class='grid'>")
        for f in ["labels.jpg", "results.png", "results.csv"]:
            src = run_dir / f
            if src.exists():
                dest = report_dir / src.name
                shutil.copy(src, dest)
                if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    uri = image_to_data_uri(src)
                    if uri:
                        html_parts.append(
                            f"<div class='card'><h3>{src.name}</h3><img src='{uri}' alt='{src.name}' style='max-width:100%'/></div>"
                        )
        for p in run_dir.glob("train_batch*.jpg"):
            dest = report_dir / p.name
            shutil.copy(p, dest)
            uri = image_to_data_uri(p)
            if uri:
                html_parts.append(
                    f"<div class='card'><h3>{p.name}</h3><img src='{uri}' alt='{p.name}' style='max-width:100%'/></div>"
                )
        # Training curves from results.csv
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                html_parts.append("</div><h2>Training history (CSV)</h2>")
                html_parts.append(
                    "<pre style='overflow:auto; background:#1b263b; padding:1rem; border-radius:8px;'>"
                )
                html_parts.append(df.to_string())
                html_parts.append("</pre>")
                # Plot training curves with best value marked on each
                if HAS_MPL and len(df) > 0:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    x = df.get("epoch", range(len(df)))
                    if "epoch" not in df.columns:
                        df = df.copy()
                        df["epoch"] = list(range(len(df)))
                    x_vals = df["epoch"].values
                    col_tries = [
                        ("train/box_loss", "Box loss", "min"),
                        ("train/cls_loss", "Class loss", "min"),
                        ("metrics/mAP50(B)", "mAP50", "max"),
                        ("metrics/mAP50-95(B)", "mAP50-95", "max"),
                    ]
                    for idx, (col, title, best_mode) in enumerate(col_tries):
                        ax = axes[idx // 2, idx % 2]
                        found = (
                            col
                            if col in df.columns
                            else next(
                                (
                                    c
                                    for c in df.columns
                                    if title.lower() in c.lower()
                                    or col.split("/")[-1].split("(")[0] in c
                                ),
                                None,
                            )
                        )
                        if found is not None:
                            y_vals = df[found].values.astype(float)
                            ax.plot(x_vals, y_vals, "b-", label="Value")
                            # Mark best: min for loss, max for metric
                            if best_mode == "min":
                                best_idx = int(np.nanargmin(y_vals))
                            else:
                                best_idx = int(np.nanargmax(y_vals))
                            best_epoch = x_vals[best_idx]
                            best_val = y_vals[best_idx]
                            ax.scatter(
                                [best_epoch],
                                [best_val],
                                color="red",
                                s=120,
                                zorder=5,
                                edgecolors="white",
                                linewidths=2,
                                label=f"Best (epoch {best_epoch})",
                            )
                            ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5)
                        ax.set_title(title)
                        ax.set_xlabel("Epoch")
                        ax.legend(loc="best", fontsize=8)
                    plt.tight_layout()
                    curve_path = report_dir / "training_curves.png"
                    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    uri = image_to_data_uri(curve_path)
                    if uri:
                        html_parts.append(
                            "<h2>Training curves</h2><div class='card'><img src='"
                            + uri
                            + "' alt='Training curves' style='max-width:100%'/></div>"
                        )
            except Exception:
                pass
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    report_file = report_dir / "combined_report.html"
    report_file.write_text("".join(html_parts), encoding="utf-8")
    print(f"Combined report saved to: {report_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COMBINED model performance report (graphs + metrics for all 3 classes)"
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/combined/weights/best.pt",
        help="Combined model weights",
    )
    parser.add_argument(
        "--data",
        default="data/combined.yaml",
        help="Combined dataset YAML",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Combined training run directory (for results.png, labels.jpg, etc.)",
    )
    # Centralize all report artifacts under results/
    parser.add_argument(
        "--output",
        default="results/combined_report",
        help="Output directory for combined report and plots",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    root = Path.cwd()
    weights = root / args.weights
    data_yaml = root / args.data
    output_dir = root / args.output
    # Validation will save plots to project/name = output_dir
    val_plots_dir = output_dir

    if not weights.exists():
        # Try a default Ultralytics-style path just in case
        alt = Path.home() / "runs" / "detect" / "combined" / "weights" / "best.pt"
        if alt.exists():
            weights = alt
        else:
            raise FileNotFoundError(f"Weights not found: {weights}. Train combined model first (train_combined.ps1).")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_run_dir(root / "runs" / "detect", "combined")
        if not run_dir and (Path.home() / "runs" / "detect" / "combined").exists():
            run_dir = Path.home() / "runs" / "detect" / "combined"

    print("Running validation for COMBINED model to generate metrics and plots...")
    metrics = run_validation_and_plots(weights, data_yaml, val_plots_dir, imgsz=args.imgsz, batch=args.batch)
    print("Building HTML report for COMBINED model...")
    build_html_report(output_dir, metrics, run_dir, val_plots_dir)
    print("Done. Open combined_report.html in a browser to view all graphs and metrics.")


if __name__ == "__main__":
    main()


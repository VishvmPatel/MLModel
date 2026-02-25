# Pothole Detection with YOLOv8

This project trains a **YOLOv8** model to **detect and outline potholes** in images. The model draws bounding boxes around each detected pothole and can classify images (pothole vs no pothole) via the detections.

## Features

- **Detection + outlining**: Model predicts bounding boxes and the scripts draw them on images.
- **Combined 3-class model**: Detects **pothole**, **fallen_tree**, and **electric_pole** (see Combined model section).
- **Accuracy-focused**: YOLOv8 training with optional `--conf` and `--max-area` filters for fewer false positives (see **ACCURACY.md**).
- **Pothole-only** or **combined** workflows: single-class `pothole` or multi-class via `data/combined.yaml`.

## Setup

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset and annotations

You need images and **YOLO-format labels** (one `.txt` per image).

- **Label format** (one line per pothole):  
  `class_id x_center y_center width height`  
  All in **normalized coordinates (0–1)** relative to image width/height.  
  Class ID for pothole is `0`.

- **Ways to create labels**:
  - [LabelImg](https://github.com/HumanSignal/labelImg) (export as YOLO).
  - [Roboflow](https://roboflow.com/) (export YOLOv8).
  - Or use a public pothole dataset already in YOLO format.

### Step 1: Prepare dataset layout

From the project root (where `Images` and `data` live):

```bash
python prepare_dataset.py --images-dir Images --output-dir data/pothole_dataset --val-ratio 0.2
```

This will:

- Split your images into **train** (80%) and **val** (20%).
- Create `data/pothole_dataset/images/train`, `images/val`, `labels/train`, `labels/val`.
- Copy images and create **empty** `.txt` files for each image (same base name as the image).

### Step 2: Annotate

- Open `data/pothole_dataset/images/train` and `images/val` in your labeling tool.
- For each image, draw bounding boxes around potholes and save as YOLO format into the corresponding `labels/train` or `labels/val` folder (same base name as the image, e.g. `image1.jpg` → `image1.txt`).
- Each line in the `.txt`: `0 x_center y_center width height` (0 = pothole, coordinates 0–1).

### Step 3: Dataset YAML

`data/pothole.yaml` should point to your dataset. After `prepare_dataset.py`, the `path` may be updated to the absolute path of `data/pothole_dataset`. If not, set it manually:

```yaml
path: data/pothole_dataset   # or full path, e.g. D:/Projects/MLModel/data/pothole_dataset
train: images/train
val: images/val
names:
  0: pothole
nc: 1
```

## Training (maximize accuracy)

- **Larger model** = better accuracy, slower: use `--model l` or `--model x`.
- **More epochs**: default is 200; increase (e.g. 300) if validation loss still improving.
- **Larger image size** for small potholes: e.g. `--imgsz 1280` (slower, more VRAM).

From project root:

```bash
python train.py --data data/pothole.yaml --model l --epochs 200 --imgsz 640 --batch 16
```

- **GPU**: leave `--device` empty for auto, or set `--device 0`.
- **CPU only**: `--device cpu` and use a small batch (e.g. `--batch 4`).

Best weights are saved to:

`runs/detect/pothole/weights/best.pt`

## Detection and outlining

Run inference so that **potholes are detected and outlined** (bounding boxes drawn) on images:

```bash
python detect.py --weights runs/detect/pothole/weights/best.pt --source Images --output-dir runs/detect/pothole_predict
```

- `--source`: image file, folder of images, or video.
- Output images with boxes are written to `--output-dir` (default: `runs/detect/pothole_predict`).
- Adjust sensitivity: `--conf 0.2` (more detections) or `--conf 0.4` (fewer, more confident).
- For fewer false positives (e.g. in an app): `--conf 0.45 --max-area 0.3` (see **ACCURACY.md**).

## View all performance graphs and metrics

To generate a single report with **all graphs and metrics** (validation metrics table, confusion matrix, F1/P/R/PR curves, training curves, label distribution, batch samples):

```bash
python report_performance.py --weights runs/detect/pothole/weights/best.pt --output runs/detect/pothole_report
```

Then open **`runs/detect/pothole_report/report.html`** in a browser. The report includes:

- **Validation metrics**: mAP50, mAP50-95, precision, recall, fitness
- **Validation plots**: confusion matrix, F1 curve, precision curve, recall curve, PR curve
- **Training run**: label distribution (`labels.jpg`), sample batches (`train_batch*.jpg`), training curves (if `results.csv` exists), and raw CSV

The model **detects and outlines** each pothole; “classification” here means the model assigns the class “pothole” to each box. Images with at least one detection can be considered “contains pothole.”

## Project structure (after setup)

```
MLModel/
├── Images/                 # Your raw images
├── data/
│   ├── pothole.yaml        # Pothole-only dataset config
│   ├── combined.yaml       # Combined 3-class config
│   ├── pothole_dataset/    # Created by prepare_dataset.py / merge_pothole_dataset.py
│   └── combined_dataset/   # Created by prepare_combined_dataset.py
│       ├── images/train, images/val
│       └── labels/train, labels/val
├── runs/detect/
│   ├── pothole/weights/    # Pothole-only model
│   ├── combined/weights/   # Combined model (best.pt, last.pt)
│   │   ├── best.pt
│   │   └── results.csv
│   ├── combined_predict/   # Detection output
│   └── single_test/       # test_single_image / test_image_verbose output
├── prepare_dataset.py, prepare_combined_dataset.py, merge_pothole_dataset.py
├── train.py, detect.py
├── add_images_to_training.py, test_image_verbose.py, fix_labelImg_canvas.py
├── report_performance.py, print_metrics.py
├── ACCURACY.md, IMPROVE_MODEL.md, LABEL_FALLEN_TREES.md
├── requirements.txt
└── README.md
```

## Quick run (all steps)

To run the full pipeline once (with minimal placeholder labels so training runs):

```bash
pip install -r requirements.txt
python prepare_dataset.py --images-dir Images --output-dir data/pothole_dataset
python create_minimal_labels.py    # optional: adds placeholder labels so training can run
python train.py --data data/pothole.yaml --model n --epochs 30 --batch 8
python detect.py --source Images --output-dir runs/detect/pothole_predict
```

For **best accuracy**, use real annotations: annotate in `data/pothole_dataset/labels/` (YOLO format) or run `python download_pothole_dataset.py` (if the public dataset download succeeds) before training. Then train with more epochs and a larger model: `python train.py --model l --epochs 200`.

Trained weights are saved to `runs/detect/pothole/weights/best.pt` (or, on some setups, under your user directory). A copy is kept in the project for `detect.py` default path.

## Combined model: pothole + fallen_tree + electric_pole

One model **detects and classifies** all three: **pothole**, **fallen tree**, and **electric pole** (with bounding box outlining for each).

### Merge a "Pothole Dataset" folder (images + labels)

If you have a folder (e.g. **Pothole Dataset**) with paired `.jpg` and `.txt` (YOLO labels), merge it into the pothole dataset:

```bash
python merge_pothole_dataset.py --source "Pothole Dataset"
```

Then run `prepare_combined_dataset.py` and train (see below).

### Prepare and train combined model

1. **Prepare combined dataset** (uses `data/pothole_dataset`, `Images_FallenTrees`, and `Images_ElectricPole`):

   ```bash
   python prepare_combined_dataset.py
   ```

   This creates `data/combined_dataset/` with classes `0: pothole`, `1: fallen_tree`, `2: electric_pole`. Refine labels in LabelImg (see **LABEL_FALLEN_TREES.md**) for better fallen_tree/electric_pole detection.

2. **Train the combined model**:

   ```bash
   python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0
   ```
   Or `.\train_combined.ps1` (Windows). To **resume** after pausing: `.\resume_combined.ps1` or add `--resume` to the command.

3. **Detect and outline all three classes** on any folder:

   ```bash
   .\detect_combined.ps1 Images
   .\detect_combined.ps1 Images_FallenTrees
   .\detect_combined.ps1 Images_ElectricPole
   ```
   Or use `detect.py` with `--weights runs/detect/combined/weights/best.pt`. Output: `runs/detect/combined_predict/`.

### Test on a single image

- **Quick test:** `.\test_single_image.ps1 "path\to\image.jpg"`
- **See all detections (low conf):** `python test_image_verbose.py "path\to\image.jpg"`
- Output is saved to **`runs/detect/single_test/`**.

### Improve the model

- **Add your own images to training:** `python add_images_to_training.py "path\to\image.jpg"` (uses current model's predictions as labels). Then run `prepare_combined_dataset.py` and re-train.
- **Fix fallen tree / electric pole labels:** See **LABEL_FALLEN_TREES.md** (draw boxes in LabelImg, class fallen_tree or electric_pole).
- **Fewer false positives in app:** See **ACCURACY.md** (use `--conf 0.45`, `--max-area 0.3`).
- **More tips:** **IMPROVE_MODEL.md**.

### Performance metrics (combined model)

- **Per-epoch metrics:** `runs/detect/combined/results.csv` (precision, recall, mAP50, mAP50-95, losses).
- **Full report (plots + metrics):**
  ```bash
  python report_performance.py --weights runs/detect/combined/weights/best.pt --data data/combined.yaml --output runs/detect/combined_report
  ```
  Open **`runs/detect/combined_report/report.html`**.
- **Print metrics to console and file:** `python print_metrics.py` → saves to `runs/detect/combined_metrics.txt`.

### LabelImg (if it crashes when drawing)

Run once: `python fix_labelImg_canvas.py`, then run `python -m labelImg`.

Output images show bounding boxes and **class labels** (pothole / fallen_tree / electric_pole). The model classifies each detection so that each image’s contents are labeled correctly.

## Quick checklist

**Pothole-only:**  
1. Put images in `Images/`.  
2. Run `python prepare_dataset.py`.  
3. Annotate potholes (YOLO format) in `data/pothole_dataset/labels/`, or run `create_minimal_labels.py` for a quick test.  
4. Run `python train.py` (optionally with `--model x --epochs 300` for higher accuracy).  
5. Run `python detect.py` to detect and outline potholes on new images/videos.

**Combined (pothole + fallen_tree + electric_pole):**  
1. Optionally merge a "Pothole Dataset" folder: `python merge_pothole_dataset.py --source "Pothole Dataset"`.  
2. Run `python prepare_combined_dataset.py`.  
3. Refine labels in `data/combined_dataset/labels/train` (LabelImg; see **LABEL_FALLEN_TREES.md**).  
4. Train: `python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined` (resume: `.\resume_combined.ps1`).  
5. Detect: `.\detect_combined.ps1 Images` or use `detect.py` with `--weights runs/detect/combined/weights/best.pt`.  
6. Metrics: `python report_performance.py --weights runs/detect/combined/weights/best.pt --data data/combined.yaml --output runs/detect/combined_report` → open `report.html`.

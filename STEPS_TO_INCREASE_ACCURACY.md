# Steps to Increase Model Accuracy

Your current **model accuracy (mAP50-95) is 77.4%**. The main limit is **pothole** detection (mAP50 ~54%, recall ~46%). Fallen tree and electric_pole are already strong (~98–99%). Follow these steps in order for the biggest gain.

---

## Step 1: Improve pothole labels (highest impact)

Bad or loose pothole boxes pull down mAP and recall.

- Open **LabelImg** (e.g. `python run_labelImg.py`).
- **Open Dir:** `data/combined_dataset/images/train`  
  **Save Dir:** `data/combined_dataset/labels/train`  
  **Format:** YOLO.
- For **pothole** images (e.g. `pothole_*.jpg`):
  - Draw **one tight box per pothole** (no big road regions, shadows, or cracks).
  - Remove or fix wrong labels.
- Do this for at least **20–50** pothole images. More is better.

**Do not** run `prepare_combined_dataset.py` after editing labels, or it will overwrite them with placeholders.

---

## Step 2: Replace placeholder labels for fallen_tree and electric_pole

Many of these still have a single “center” box. Real boxes help the model learn position and size.

- In LabelImg, open images under `data/combined_dataset/images/train` (e.g. `fallentree_*.jpg`, `electricpole_*.jpg`).
- For each image, **delete the placeholder** and draw **one box around the whole object** (tree or pole).
- Use class **fallen_tree** (1) or **electric_pole** (2).
- Even 15–30 images per class helps.

---

## Step 3: Add more pothole images

Add images the model missed or got wrong:

```powershell
cd d:\Projects\MLModel
python add_images_to_training.py "path\to\pothole_image.jpg"
```

Then **rebuild** the combined dataset and **re-prepare** (only if you added new images):

```powershell
python prepare_combined_dataset.py
```

Then **label the new images** in LabelImg (Step 1). If you only fixed existing labels, skip `prepare_combined_dataset.py` and go to Step 4.

---

## Step 4: Train longer and/or use a larger model

- **More epochs** (e.g. 80–100) if validation mAP was still improving at the end.
- **Larger model** for better accuracy (needs more GPU memory):

| Model | Command       | Accuracy | Speed / VRAM   |
|-------|---------------|----------|----------------|
| nano (current) | `--model n` | baseline | fastest        |
| small | `--model s`   | better   | moderate       |
| medium | `--model m`  | best     | slower, more VRAM |

Example (small model, 80 epochs):

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0
```

To **resume** from the last run:

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0 --resume
```

---

## Step 5: Optional – larger input size (small potholes)

If potholes are small in the image:

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --imgsz 1280 --batch 4 --name combined --device 0
```

`--imgsz 1280` uses more VRAM; reduce `--batch` if you get out-of-memory errors.

---

## Step 6: Re-run metrics and report

After re-training:

```powershell
python print_metrics.py
python report_performance_combined.py
```

Check **accuracy_percent.txt** and **combined_report.html** for the new mAP50-95.

---

## Quick checklist

| Priority | Action |
|----------|--------|
| 1 | Fix and tighten **pothole** labels in LabelImg (20–50+ images). |
| 2 | Replace placeholder boxes with **real boxes** for fallen_tree and electric_pole (15–30 per class). |
| 3 | Add **new pothole images** with `add_images_to_training.py`, then `prepare_combined_dataset.py`, then label them. |
| 4 | Re-train with **more epochs** (80–100) and/or **larger model** (`--model s` or `m`). |
| 5 | Optionally use **`--imgsz 1280`** if potholes are small. |

Focus on **Steps 1 and 2** first; better labels usually give the largest accuracy gain.

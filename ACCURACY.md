# Getting accurate predictions for your app

If the model is labeling non-potholes (buildings, signs, shadows, large road areas) as potholes, or drawing huge boxes, use these steps so the app gets reliable results.

## 1. Use a higher confidence threshold

**In your app or when running detection**, use a **higher confidence** so only confident detections are shown:

- **Default** `--conf 0.25`: more detections, more false positives.
- **Recommended for apps** `--conf 0.45` or `--conf 0.5`: fewer but more reliable detections.

Example:

```bash
python detect.py --weights runs/detect/combined/weights/best.pt --source Images --output-dir runs/detect/combined_predict --conf 0.45
```

`detect_combined.ps1` already uses `--conf 0.45` by default.

## 2. Filter out oversized boxes (potholes)

Potholes are usually **small**. Very large “pothole” boxes often mean buildings or big road regions. You can **drop boxes that cover too much of the image**:

```bash
python detect.py --weights runs/detect/combined/weights/best.pt --source Images --output-dir runs/detect/combined_predict --conf 0.45 --max-area 0.3
```

- `--max-area 0.3`: ignore any box larger than 30% of the image area.
- Use `0.25` for stricter filtering, or omit `--max-area` if you don’t want this filter.

In the app, you can do the same: keep only boxes where `(width * height) / (image_width * image_height) <= 0.3` (or your chosen ratio).

## 3. Improve training labels (biggest impact)

Wrong or loose labels in the dataset cause wrong predictions:

- **Pothole**: Draw boxes **only** around real potholes, and **tight** (not huge regions). Remove labels on shadows, cracks, or non-potholes.
- **Fallen tree / electric pole**: Replace placeholder (center) labels with real boxes when you can.

Ways to improve:

- Manually correct in **LabelImg** (YOLO format) in `data/combined_dataset/labels/`.
- If you used `auto_annotate` with a large `--expand`, re-run with a smaller expand or fix boxes by hand, then **re-train**.

After fixing labels, re-train:

```bash
python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device 0
```

## 4. Train longer and tune

- **More epochs** (e.g. 80–100) can help if validation loss was still improving.
- **Larger image size** for small objects: `--imgsz 1280` (slower, more VRAM).
- **Larger model** for accuracy: `--model s` or `--model m` instead of `n`.

## Summary for the app

1. Use **`--conf 0.45`** (or 0.5) at inference.
2. Use **`--max-area 0.3`** (or 0.25) to drop huge “pothole” boxes.
3. Improve **pothole (and other) labels** in the dataset and **re-train**.
4. Optionally train longer or with larger `imgsz` / `model`.

Then run detection (or your app) with the same `conf` and `max_area` so the model doesn’t “prove you wrong” with obvious false positives.

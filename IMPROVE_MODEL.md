# Improve the model (higher confidence, fewer misses)

If the model **detects** a pothole but with **low confidence** (e.g. 0.23), or **misses** some images, add those images to training and re-train.

## 1. Add your images to the pothole training set

Use the current model to generate labels, then add image + labels to the dataset:

```powershell
cd d:\Projects\MLModel
python add_images_to_training.py "path\to\your\pothole_image.jpg"
```

**Examples:**

- Single image:
  ```powershell
  python add_images_to_training.py "D:\Downloads\pothole_closeup.jpg"
  ```
- Multiple images:
  ```powershell
  python add_images_to_training.py "img1.jpg" "img2.png"
  ```
- All images in a folder:
  ```powershell
  python add_images_to_training.py "D:\MyPotholePhotos"
  ```

The script will:
- Run the current model on each image (with low conf threshold so detections are kept)
- Save the predicted boxes as YOLO labels
- Copy images into `data/pothole_dataset/images/train`

You can refine the generated labels in **LabelImg** (edit the `.txt` files in `data/pothole_dataset/labels/train`) before re-training.

## 2. Rebuild the combined dataset and re-train

```powershell
python prepare_combined_dataset.py
python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device 0
```

To **resume** from a previous run instead of starting from scratch:

```powershell
python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device 0 --resume
```

## 3. Optional: train longer or use a larger model

- **More epochs** (e.g. 80): better convergence, often higher confidence.
- **Larger model**: `--model s` or `--model m` for better accuracy (slower, more VRAM).
- **Larger image size**: `--imgsz 1280` if you have small potholes and enough GPU memory.

Example:

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0
```

## Summary

1. **Add images** that the model missed or predicted with low confidence:  
   `python add_images_to_training.py "your_image.jpg"`
2. **Rebuild dataset**: `python prepare_combined_dataset.py`
3. **Re-train**: `python train.py --data data/combined.yaml ... --name combined --device 0`
4. Test again with `.\test_single_image.ps1 "your_image.jpg"` — confidence should improve.

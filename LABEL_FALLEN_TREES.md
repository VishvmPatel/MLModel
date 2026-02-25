# Fix fallen tree detection (wrong class + small box)

If the model labels a **fallen tree** as "pothole" or draws a **tiny box** on a few leaves instead of the whole tree, it's because fallen trees were trained with **placeholder** labels (one small center box). Fix it by adding **real labels** that cover the whole tree and use class **fallen_tree**.

## Step 1: Add your tree image to the dataset

1. Copy your image (e.g. the one with the tree across the road) into **`Images_FallenTrees`**.
2. Rebuild the combined dataset:
   ```powershell
   cd d:\Projects\MLModel
   python prepare_combined_dataset.py
   ```
3. The image will appear in **`data/combined_dataset/images/train`** (or `val`) with a name like **`fallentree_012345.png`**. Note that filename.

## Step 2: Label the whole tree in LabelImg

1. Install LabelImg if needed: `pip install labelImg`, then run `labelImg`.
2. In LabelImg:
   - **Open Dir**: `d:\Projects\MLModel\data\combined_dataset\images\train`
   - **Change Save Dir**: `d:\Projects\MLModel\data\combined_dataset\labels\train`
   - **Format**: YOLO (not PascalVOC).
3. Open the **fallen tree image** (e.g. `fallentree_012345.png`).
4. Draw **one box around the entire fallen tree** (trunk + branches + leaves that belong to it). Make it as tight as you can but cover the full tree.
5. When asked for the class, choose **fallen_tree** (or class **1** in YOLO: `1`).
6. Save. This overwrites the placeholder label with your correct box.

Do this for **several** fallen tree images (at least 10–20) so the model learns the class well.

## Step 3: Do NOT re-run prepare_combined_dataset after labeling

Re-running `prepare_combined_dataset.py` overwrites labels with placeholders again. So:

- Add new images → run `prepare_combined_dataset.py` → then label in LabelImg.
- If you only fixed labels (no new images), skip `prepare_combined_dataset.py` and go to Step 4.

## Step 4: Re-train the combined model

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0
```

Or resume if you had paused:

```powershell
python train.py --data data/combined.yaml --model s --epochs 80 --batch 8 --name combined --device 0 --resume
```

## Step 5: Test again

Run detection on the same tree image:

```powershell
python test_image_verbose.py "path\to\your\tree_image.png"
```

You should see **"fallen_tree"** (not pothole) and a box around the whole tree.

---

**Summary:** The model mislabels and uses small boxes because it was trained with placeholder fallen_tree labels. Draw **one box around the whole tree**, class **fallen_tree (1)**, in `data/combined_dataset/labels/train`, then re-train. Label multiple tree images for best results.

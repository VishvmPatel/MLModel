"""
Prepare combined YOLOv8 dataset for pothole + fallen_tree + electric_pole detection.
- Merges pothole_dataset (class 0), Images_FallenTrees (class 1), Images_ElectricPole (class 2).
- Uses unique filenames (pothole_*, fallentree_*, electricpole_*) to avoid collisions.
- Non-pothole images get a minimal placeholder label so training can run;
  replace with real annotations or use auto_annotate after training a first model.
"""

import random
import shutil
from pathlib import Path

# Classes: 0 pothole, 1 fallen_tree, 2 electric_pole
POTHOLE_CLASS = 0
FALLENTREE_CLASS = 1
ELECTRICPOLE_CLASS = 2
PLACEHOLDER_BOX = "0.5 0.5 0.2 0.2\n"  # center, 20% size

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(folder: Path) -> list[Path]:
    return [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS]


def prepare_combined_dataset(
    pothole_root: str = "data/pothole_dataset",
    fallen_trees_dir: str = "Images_FallenTrees",
    electric_pole_dir: str = "Images_ElectricPole",
    output_dir: str = "data/combined_dataset",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    pothole = Path(pothole_root)
    fallen = Path(fallen_trees_dir)
    electric = Path(electric_pole_dir)
    out = Path(output_dir)
    if not pothole.exists():
        raise FileNotFoundError(f"Pothole dataset not found: {pothole}")
    if not fallen.exists():
        raise FileNotFoundError(f"Fallen trees images not found: {fallen}")
    if not electric.exists():
        raise FileNotFoundError(f"Electric pole images not found: {electric}")

    fallen_images = _collect_images(fallen)
    electric_images = _collect_images(electric)
    if not fallen_images:
        raise FileNotFoundError(f"No images found in {fallen}")
    if not electric_images:
        raise FileNotFoundError(f"No images found in {electric}")

    random.seed(seed)
    random.shuffle(fallen_images)
    random.shuffle(electric_images)
    n_val_ft = max(1, int(len(fallen_images) * val_ratio))
    n_val_ep = max(1, int(len(electric_images) * val_ratio))
    val_fallen = set(fallen_images[:n_val_ft])
    train_fallen = [f for f in fallen_images if f not in val_fallen]
    val_electric = set(electric_images[:n_val_ep])
    train_electric = [f for f in electric_images if f not in val_electric]

    for split, fallen_list, electric_list in [
        ("train", train_fallen, train_electric),
        ("val", list(val_fallen), list(val_electric)),
    ]:
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # 1) Copy pothole images and labels (class 0)
        p_img = pothole / "images" / split
        p_lbl = pothole / "labels" / split
        if p_img.exists():
            for img_path in p_img.iterdir():
                if img_path.suffix.lower() not in EXTENSIONS:
                    continue
                dest_name = f"pothole_{img_path.name}"
                dest_img = img_dir / dest_name
                if not dest_img.exists() or dest_img.stat().st_mtime < img_path.stat().st_mtime:
                    shutil.copy2(img_path, dest_img)
                lbl_src = p_lbl / (img_path.stem + ".txt")
                lbl_dest = lbl_dir / (Path(dest_name).stem + ".txt")
                if lbl_src.exists() and lbl_src.stat().st_size > 0:
                    shutil.copy2(lbl_src, lbl_dest)
                else:
                    lbl_dest.write_text(f"{POTHOLE_CLASS} {PLACEHOLDER_BOX}", encoding="utf-8")

        # 2) Copy fallen tree images, placeholder labels (class 1)
        for i, img_path in enumerate(fallen_list):
            dest_name = f"fallentree_{i:06d}{img_path.suffix}"
            dest_img = img_dir / dest_name
            if not dest_img.exists() or dest_img.stat().st_mtime < img_path.stat().st_mtime:
                shutil.copy2(img_path, dest_img)
            lbl_dest = lbl_dir / (Path(dest_name).stem + ".txt")
            lbl_dest.write_text(f"{FALLENTREE_CLASS} {PLACEHOLDER_BOX}", encoding="utf-8")

        # 3) Copy electric pole images, placeholder labels (class 2)
        for i, img_path in enumerate(electric_list):
            dest_name = f"electricpole_{i:06d}{img_path.suffix}"
            dest_img = img_dir / dest_name
            if not dest_img.exists() or dest_img.stat().st_mtime < img_path.stat().st_mtime:
                shutil.copy2(img_path, dest_img)
            lbl_dest = lbl_dir / (Path(dest_name).stem + ".txt")
            lbl_dest.write_text(f"{ELECTRICPOLE_CLASS} {PLACEHOLDER_BOX}", encoding="utf-8")

    # Update YAML path and ensure 3 classes
    yaml_path = Path("data/combined.yaml")
    abs_path = out.resolve().as_posix()
    yaml_content = (
        f"# Combined: pothole (0) + fallen_tree (1) + electric_pole (2)\n"
        f"path: {abs_path}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: pothole\n  1: fallen_tree\n  2: electric_pole\nnc: 3\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    n_train = len([f for f in (out / "images" / "train").iterdir() if f.suffix.lower() in EXTENSIONS])
    n_val = len([f for f in (out / "images" / "val").iterdir() if f.suffix.lower() in EXTENSIONS])
    print(f"Combined dataset: {output_dir}")
    print(f"  Train: {n_train} images -> {out / 'images' / 'train'}")
    print(f"  Val:   {n_val} images -> {out / 'images' / 'val'}")
    print("  Classes: 0=pothole, 1=fallen_tree, 2=electric_pole. Refine labels or auto_annotate after first training.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepare combined pothole + fallen_tree + electric_pole dataset")
    p.add_argument("--pothole", default="data/pothole_dataset", help="Pothole dataset root")
    p.add_argument("--fallen-trees", default="Images_FallenTrees", help="Folder of fallen tree images")
    p.add_argument("--electric-pole", default="Images_ElectricPole", help="Folder of electric pole images")
    p.add_argument("--output", default="data/combined_dataset", help="Output dataset root")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_combined_dataset(
        pothole_root=args.pothole,
        fallen_trees_dir=args.fallen_trees,
        electric_pole_dir=args.electric_pole,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

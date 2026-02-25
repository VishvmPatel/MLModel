# Train combined model: pothole + fallen_tree (outlining both).
# Run after: python prepare_combined_dataset.py
# Usage: .\train_combined.ps1
# Optional: add --resume to continue from last.pt

Set-Location $PSScriptRoot
$device = "0"  # GPU; use "" for CPU
python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device $device

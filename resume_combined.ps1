# Resume combined model training (pothole + fallen_tree + electric_pole). Run this tomorrow after pausing today.
# Usage: .\resume_combined.ps1

Set-Location $PSScriptRoot
python train.py --data data/combined.yaml --model n --epochs 50 --batch 8 --name combined --device 0 --resume

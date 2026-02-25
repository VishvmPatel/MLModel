# Resume pothole training on GPU. Run this in a terminal and leave it open.
# Usage: .\resume_training.ps1

Set-Location $PSScriptRoot
python train.py --data data/pothole.yaml --model n --epochs 50 --batch 8 --name pothole --device 0 --resume

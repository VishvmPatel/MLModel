# Run this AFTER the pip torch install finishes (the 2.6GB download).
# 1. Install torchvision with CUDA
# 2. Verify GPU
# 3. Resume training on GPU

Write-Host "Installing torchvision with CUDA..."
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124

Write-Host "`nChecking GPU..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`nResuming training on GPU..."
Set-Location $PSScriptRoot
python train.py --data data/pothole.yaml --model n --epochs 50 --batch 8 --name pothole --resume --device 0

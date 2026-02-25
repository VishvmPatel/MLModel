# Run combined model (pothole + fallen_tree + electric_pole): detect and outline all three classes.
# Usage: .\detect_combined.ps1 [source_folder]
# Example: .\detect_combined.ps1 Images
#          .\detect_combined.ps1 Images_FallenTrees
#          .\detect_combined.ps1 Images_ElectricPole

param([string]$Source = "Images")
Set-Location $PSScriptRoot
$weights = "runs\detect\combined\weights\best.pt"
if (-not (Test-Path $weights)) {
    Write-Host "Weights not found: $weights. Train first: .\train_combined.ps1"
    exit 1
}
# Use higher conf (0.45) and max-area (0.3) for fewer false positives; drop for more detections
python detect.py --weights $weights --source $Source --output-dir runs/detect/combined_predict --conf 0.45 --max-area 0.3 --device 0
Write-Host "Predictions (pothole + fallen_tree + electric_pole) saved to runs/detect/combined_predict"

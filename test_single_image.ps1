# Test the combined model on ONE image with relaxed settings so you can see if it detects anything.
# Use this to verify the model (e.g. clear pothole photo). No max-area filter; lower conf so detections show.
# Usage: .\test_single_image.ps1 "path\to\your\image.jpg"
# Example: .\test_single_image.ps1 "D:\Downloads\pothole.jpg"
#          .\test_single_image.ps1 "test_image.jpg"

param([Parameter(Mandatory=$true)][string]$ImagePath)
Set-Location $PSScriptRoot
$weights = "runs\detect\combined\weights\best.pt"
if (-not (Test-Path $weights)) {
    Write-Host "Weights not found: $weights. Train first: .\train_combined.ps1"
    exit 1
}
if (-not (Test-Path $ImagePath)) {
    Write-Host "Image not found: $ImagePath"
    exit 1
}
# Relaxed: conf 0.2, no max-area filter - so you can see any detection the model makes
python detect.py --weights $weights --source $ImagePath --output-dir runs/detect/single_test --conf 0.2 --device 0
$outDir = "runs\detect\single_test"
$baseName = [System.IO.Path]::GetFileName($ImagePath)
Write-Host "Output saved to: $outDir\$baseName"
Write-Host "Open that image to see if the model detected the pothole / fallen tree / electric pole."

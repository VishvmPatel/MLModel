# Run this after training finishes to get predictions and performance report.
# Usage: .\run_after_training.ps1

$weights = "runs\detect\pothole\weights\best.pt"
if (-not (Test-Path $weights)) {
    $alt = "$env:USERPROFILE\runs\detect\runs\detect\pothole\weights\best.pt"
    if (Test-Path $alt) { $weights = $alt }
    else { Write-Host "No best.pt found. Train first: python train.py"; exit 1 }
}

Write-Host "Running detection on Images..."
python detect.py --weights $weights --source Images --output-dir runs/detect/pothole_predict --conf 0.2

Write-Host "`nGenerating performance report..."
python report_performance.py --weights $weights --output runs/detect/pothole_report

Write-Host "`nDone. Open runs/detect/pothole_report/report.html for graphs and metrics."

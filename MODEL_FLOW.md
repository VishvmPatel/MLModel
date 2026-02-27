## Model workflow flowchart

The diagram below shows the end-to-end workflow of your object detection pipeline, from data collection to saving results and reports.

```mermaid
flowchart LR
    A[Collect raw images<br/>(potholes, fallen trees, poles)] --> B[Annotate images<br/>with bounding boxes]
    B --> C[Prepare YOLO dataset<br/>(train/val split, YAML)]
    C --> D[Train YOLOv8 model<br/>with train.py]
    D --> E[Validate trained model<br/>(metrics, curves, confusion matrix)]
    E --> F[Generate reports & graphs<br/>with report_performance.py<br/>→ saved under results/]
    D --> G[Run inference on new images<br/>with detect.py]
    G --> H[Review detections & images<br/>and iterate on dataset/training]
```


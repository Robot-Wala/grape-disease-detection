import os
from ultralytics import YOLO

ckpt = "runs/detect/train/weights/last.pt"
resume_exists = os.path.exists(ckpt)

model = YOLO(ckpt if resume_exists else "yolov8s.pt")

print(f"--- {'Resuming' if resume_exists else 'Starting fresh'} ---")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.0005,
    patience=25,
    device=0, 
    resume=resume_exists
)
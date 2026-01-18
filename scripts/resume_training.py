from ultralytics import YOLO
import torch
import os

print("="*60)
print("ARCHCAD TRAINING - CRASH-PROOF")
print("="*60)

checkpoint_path = 'models/archcad_detector/weights/last.pt'

if os.path.exists(checkpoint_path):
    print(f"\n✓ Resuming from: {checkpoint_path}")
    model = YOLO(checkpoint_path)
else:
    print("\n✓ Starting fresh training...")
    model = YOLO('yolov8n.pt')

print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("\n📊 Batch size: 4 (reduced to prevent OOM)")
print("⏱️  Estimated: 3-4 hours\n")

results = model.train(
    data='archcad_yolo/data.yaml',
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    amp=True,
    cache=False,
    optimizer='AdamW',
    lr0=0.001,
    patience=15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    flipud=0.5,
    fliplr=0.5,
    project='models',
    name='archcad_detector',
    exist_ok=True,
    verbose=True,
    resume=True,
    val=True,
    save=True,
    save_period=5,
    plots=True
)

print("\n✅ TRAINING COMPLETE!")
print(f"📁 Best model: models/archcad_detector/weights/best.pt")

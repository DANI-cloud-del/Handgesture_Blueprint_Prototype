from ultralytics import YOLO
import torch

print("="*60)
print("ARCHCAD MODEL TRAINING (4GB GPU OPTIMIZED)")
print("="*60)

# Check GPU
print(f"\n🖥️  Device: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load pretrained YOLOv8 nano (smallest model for 4GB GPU)
model = YOLO('yolov8n.pt')

print("\n📚 Starting training...")
print("⏱️  Estimated time: 2-3 hours on RTX 3050")

# Train with 4GB GPU optimized settings
results = model.train(
    data='archcad_yolo/data.yaml',
    
    # Model settings
    epochs=50,              # 50 epochs for good learning
    imgsz=640,              # Standard image size
    
    # Batch size - CRITICAL for 4GB GPU
    batch=8,                # Increased from 4 (you have ~5K samples)
    
    # Hardware settings
    device=0,               # Use GPU 0
    workers=4,              # Use 4 CPU cores
    
    # Memory optimization
    amp=True,               # Automatic Mixed Precision (saves VRAM)
    cache=False,            # Don't cache images (saves RAM)
    
    # Training optimization
    optimizer='AdamW',
    lr0=0.001,             # Learning rate
    patience=15,            # Early stopping after 15 epochs no improvement
    
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    flipud=0.5,            # Vertical flip (blueprints can be rotated)
    fliplr=0.5,            # Horizontal flip
    
    # Output
    project='models',
    name='archcad_detector',
    exist_ok=True,
    verbose=True,
    
    # Validation
    val=True,
    save=True,
    save_period=10,         # Save checkpoint every 10 epochs
    plots=True              # Generate training plots
)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"📁 Model saved to: models/archcad_detector/weights/best.pt")
print(f"\n📊 Final Results:")
print(f"  - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

print("\n✅ Next steps:")
print("   1. Test model: python scripts/test_model.py")
print("   2. View training plots: models/archcad_detector/")

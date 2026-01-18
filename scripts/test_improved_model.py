from ultralytics import YOLO
import cv2

print("="*60)
print("TEST IMPROVED MODEL - NEARBY DOORS FIX")
print("="*60)

# Load best model
model = YOLO('models/archcad_detector/weights/best.pt')

print("\n✓ Model loaded successfully")
print(f"✓ Classes: {model.names}")

# Test image path
test_image = 'path/to/your/blueprint_with_adjacent_doors.png'

print(f"\n🔍 Testing on: {test_image}")

# Test with different NMS settings
test_configs = [
    {'name': 'Default', 'iou': 0.5, 'conf': 0.25},
    {'name': 'Relaxed', 'iou': 0.3, 'conf': 0.2},
    {'name': 'Very Relaxed', 'iou': 0.2, 'conf': 0.15},
]

print("\n📊 Testing different NMS configurations:\n")

for config in test_configs:
    # Run detection
    results = model.predict(
        test_image,
        iou=config['iou'],
        conf=config['conf'],
        max_det=300,
        imgsz=1024,  # Larger image for small objects
        verbose=False
    )
    
    # Count detections by class
    detections = {}
    for box in results[0].boxes:
        cls = model.names[int(box.cls[0])]
        detections[cls] = detections.get(cls, 0) + 1
    
    print(f"{config['name']:15} (iou={config['iou']}, conf={config['conf']}):")
    print(f"  Doors: {detections.get('door', 0)}")
    print(f"  Windows: {detections.get('window', 0)}")
    print(f"  Total: {sum(detections.values())}")
    print()
    
    # Save visualization
    output_path = f"test_results_{config['name'].lower().replace(' ', '_')}.jpg"
    
    # Draw boxes
    img = cv2.imread(test_image)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0]
        cls = int(box.cls[0])
        
        # Color code by class
        colors = {
            'door': (0, 255, 0),
            'window': (255, 0, 0),
            'wall': (128, 128, 128)
        }
        color = colors.get(model.names[cls], (255, 255, 0))
        
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, img)
    print(f"  ✓ Saved: {output_path}")

print("\n✅ Testing complete!")
print("   Compare the outputs to see which NMS setting works best")

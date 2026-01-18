from ultralytics import YOLO
import os
import glob

print("="*60)
print("BATCH BLUEPRINT TESTING")
print("="*60)

# Load trained model
model = YOLO('models/archcad_detector/weights/best.pt')

# Find all blueprint images
blueprint_dir = 'static/blueprints/'
image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']

blueprint_files = []
for pattern in image_patterns:
    blueprint_files.extend(glob.glob(os.path.join(blueprint_dir, pattern)))

if not blueprint_files:
    print(f"\n❌ No blueprint files found in {blueprint_dir}")
    exit(1)

print(f"\n✓ Found {len(blueprint_files)} blueprint(s)")
print()

# Test each blueprint
all_results = {}

for i, blueprint_path in enumerate(blueprint_files, 1):
    filename = os.path.basename(blueprint_path)
    
    print(f"[{i}/{len(blueprint_files)}] Processing: {filename}")
    
    # Run detection
    results = model.predict(
        blueprint_path,
        conf=0.25,        # Confidence threshold
        save=True,        # Save annotated images
        show_labels=True,
        show_conf=True,
        project='runs/detect',
        name=f'test_{i}',
        exist_ok=True
    )
    
    # Count detections
    detections = {}
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            detections[class_name] = detections.get(class_name, 0) + 1
    
    all_results[filename] = detections
    
    # Print summary
    total = sum(detections.values())
    print(f"  ✓ Detected {total} objects:")
    for name, count in sorted(detections.items()):
        print(f"    - {name}: {count}")
    print()

# Overall summary
print("="*60)
print("SUMMARY - ALL BLUEPRINTS")
print("="*60)

for filename, detections in all_results.items():
    total = sum(detections.values())
    print(f"\n{filename}: {total} objects")
    for name, count in sorted(detections.items()):
        print(f"  {name:15s}: {count:3d}")

print("\n" + "="*60)
print(f"✓ Annotated images saved to: runs/detect/test_*/")
print("="*60)

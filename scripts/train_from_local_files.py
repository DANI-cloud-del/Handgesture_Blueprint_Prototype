import os
import json
import zipfile
from PIL import Image
from tqdm import tqdm
import random
import shutil

print("=" * 60)
print("TRAIN FROM LOCAL ARCHCAD FILES")
print("=" * 60)

# Configuration
PNG_ZIP = input("\nEnter path to png.zip file: ").strip()
JSON_ZIP = input("Enter path to json.zip file: ").strip()

MAX_SAMPLES = 5000
VAL_SPLIT = 0.2
OUTPUT_DIR = 'archcad_yolo'

# Verify files exist
if not os.path.exists(PNG_ZIP):
    print(f"\n❌ Error: {PNG_ZIP} not found!")
    exit(1)

if not os.path.exists(JSON_ZIP):
    print(f"\n❌ Error: {JSON_ZIP} not found!")
    exit(1)

print("\n✓ Files found!")
print(f"  PNG: {PNG_ZIP}")
print(f"  JSON: {JSON_ZIP}")

# Class mapping
CLASS_MAPPING = {
    'single_door': 0, 'double_door': 0, 'sliding_door': 0, 'revolving_door': 0,
    'single_window': 1, 'double_window': 1, 'bay_window': 1,
    'table': 2, 'chair': 3, 'bed': 4, 'sofa': 5,
    'toilet': 6, 'sink': 7, 'bathtub': 8,
    'wall': 9, 'stair': 10, 'elevator': 11, 'escalator': 11,
    'parking': 12
}

CLASS_NAMES = [
    'door', 'window', 'table', 'chair', 'bed', 'sofa',
    'toilet', 'sink', 'bathtub', 'wall', 'stair', 
    'elevator', 'parking'
]

# Create output directories
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

print(f"\n📊 Configuration:")
print(f"  Max samples: {MAX_SAMPLES:,}")
print(f"  Output: {OUTPUT_DIR}/")

input("\nPress Enter to start processing...")

# Open ZIP files
print("\n🔄 Opening ZIP files...")
png_archive = zipfile.ZipFile(PNG_ZIP, 'r')
json_archive = zipfile.ZipFile(JSON_ZIP, 'r')

# Get file lists
png_files = [f for f in png_archive.namelist() if f.endswith('.png')]
json_files = [f for f in json_archive.namelist() if f.endswith('.json')]

print(f"\n✓ Found {len(png_files):,} PNG files")
print(f"✓ Found {len(json_files):,} JSON files")

# Create filename mapping
png_map = {os.path.basename(f).replace('.png', ''): f for f in png_files}
json_map = {os.path.basename(f).replace('.json', ''): f for f in json_files}

# Find matching pairs
common_ids = set(png_map.keys()) & set(json_map.keys())
print(f"\n✓ Found {len(common_ids):,} matching pairs")

# Process samples
train_count = 0
val_count = 0
processed = 0
skipped = 0

sample_ids = list(common_ids)[:MAX_SAMPLES]

print(f"\n🔄 Processing {len(sample_ids):,} samples...\n")

for sample_id in tqdm(sample_ids, desc="Processing"):
    try:
        # Read image from ZIP
        png_path = png_map[sample_id]
        img_data = png_archive.read(png_path)
        
        # Read JSON from ZIP
        json_path = json_map[sample_id]
        json_data = json.loads(json_archive.read(json_path))
        
        # Open image
        from io import BytesIO
        image = Image.open(BytesIO(img_data))
        
        # Convert to YOLO annotations
        yolo_annotations = []
        width, height = image.size
        
        if isinstance(json_data, list):
            for primitive in json_data:
                semantic = primitive.get('semantic', 'others')
                
                if semantic not in CLASS_MAPPING:
                    continue
                
                class_id = CLASS_MAPPING[semantic]
                
                # Extract bounding box
                prim_type = primitive.get('type')
                
                try:
                    if prim_type == 'LINE':
                        start = primitive['start']
                        end = primitive['end']
                        x_min = min(start[0], end[0])
                        y_min = min(start[1], end[1])
                        x_max = max(start[0], end[0])
                        y_max = max(start[1], end[1])
                        
                    elif prim_type in ['CIRCLE', 'ARC']:
                        center = primitive.get('center', [0, 0])
                        radius = primitive.get('radius', 10)
                        x_min = center[0] - radius
                        y_min = center[1] - radius
                        x_max = center[0] + radius
                        y_max = center[1] + radius
                    else:
                        continue
                    
                    # Calculate normalized YOLO format
                    x_center = ((x_min + x_max) / 2) / width
                    y_center = ((y_min + y_max) / 2) / height
                    box_width = (x_max - x_min + 10) / width
                    box_height = (y_max - y_min + 10) / height
                    
                    # Clip to valid range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    box_width = max(0, min(1, box_width))
                    box_height = max(0, min(1, box_height))
                    
                    yolo_annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                    )
                except:
                    continue
        
        # Skip if no annotations
        if not yolo_annotations:
            skipped += 1
            continue
        
        # Determine split
        is_val = random.random() < VAL_SPLIT
        split = 'val' if is_val else 'train'
        
        # Save image
        img_filename = f"archcad_{processed:06d}.jpg"
        img_path = f"{OUTPUT_DIR}/images/{split}/{img_filename}"
        
        # Resize and save
        image = image.convert('RGB').resize((640, 640), Image.LANCZOS)
        image.save(img_path, quality=85)
        
        # Save labels
        label_filename = f"archcad_{processed:06d}.txt"
        label_path = f"{OUTPUT_DIR}/labels/{split}/{label_filename}"
        
        with open(label_path, 'w') as f:
            for ann in yolo_annotations:
                f.write(f"{ann}\n")
        
        if is_val:
            val_count += 1
        else:
            train_count += 1
        
        processed += 1
        
    except Exception as e:
        skipped += 1
        continue

# Close archives
png_archive.close()
json_archive.close()

print("\n" + "=" * 60)
print("✅ PROCESSING COMPLETE!")
print("=" * 60)
print(f"✓ Train samples: {train_count:,}")
print(f"✓ Val samples:   {val_count:,}")
print(f"⚠ Skipped:       {skipped:,}")

# Create data.yaml
yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

with open(f'{OUTPUT_DIR}/data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"\n✓ Created: {OUTPUT_DIR}/data.yaml")
print("\n✅ Ready for training!")
print("   Run: python scripts/train_model.py")

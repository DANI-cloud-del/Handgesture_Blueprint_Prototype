import os
import json
import zipfile
from PIL import Image
from tqdm import tqdm
import random

print("=" * 60)
print("TRAIN FROM LOCAL ARCHCAD FILES - ALL SEMANTICS")
print("=" * 60)

# Use ALL semantic IDs found in the dataset
# Group them into our target classes based on frequency and likely meaning
SEMANTIC_TO_CLASS = {
    # Common architectural elements (based on frequency analysis)
    # High frequency IDs (likely walls, structures)
    100: 9,  # wall/structure (most common - 28,764 occurrences)
    20: 9,   # wall variant (8,524 occurrences)
    
    # Medium frequency (likely furniture, fixtures)
    0: 0,    # door type 1
    1: 0,    # door type 2  
    2: 0,    # door type 3
    
    4: 1,    # window type 1
    6: 1,    # window type 2
    
    7: 2,    # table
    9: 3,    # chair
    10: 4,   # bed/large furniture
    11: 5,   # sofa
    
    12: 6,   # toilet/bathroom fixture
    13: 7,   # sink
    14: 8,   # bathtub
    
    15: 9,   # wall/column
    16: 10,  # stair
    17: 11,  # elevator
    18: 11,  # escalator
    19: 12,  # parking
    21: 9,   # structural element
    
    # Add any other IDs as "wall" (catch-all for structural)
}

CLASS_NAMES = [
    'door', 'window', 'table', 'chair', 'bed', 'sofa',
    'toilet', 'sink', 'bathtub', 'wall', 'stair', 
    'elevator', 'parking'
]

# File paths
PNG_ZIP = '/home/dani/.cache/huggingface/hub/datasets--jackluoluo--ArchCAD/snapshots/035db66d2c5794395f63d793324b3fe3d360ffe3/data/png.zip'
JSON_ZIP = '/home/dani/.cache/huggingface/hub/datasets--jackluoluo--ArchCAD/snapshots/035db66d2c5794395f63d793324b3fe3d360ffe3/data/json.zip'

MAX_SAMPLES = 5000
VAL_SPLIT = 0.2
OUTPUT_DIR = 'archcad_yolo'
ACCEPT_ALL_SEMANTICS = True  # Accept any semantic ID, map unknown to "wall"

# Create output directories
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

print(f"\n📊 Configuration:")
print(f"  Max samples: {MAX_SAMPLES:,}")
print(f"  Accept all semantics: {ACCEPT_ALL_SEMANTICS}")
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
unknown_semantics = set()

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
        
        # Extract entities
        entities = json_data.get('entities', [])
        
        if not entities:
            skipped += 1
            continue
        
        # Convert to YOLO annotations
        yolo_annotations = []
        width, height = image.size
        
        for entity in entities:
            semantic_id = entity.get('semantic')
            
            # Map semantic ID to class
            if semantic_id in SEMANTIC_TO_CLASS:
                class_id = SEMANTIC_TO_CLASS[semantic_id]
            elif ACCEPT_ALL_SEMANTICS:
                # Unknown semantic - treat as wall (class 9)
                class_id = 9
                unknown_semantics.add(semantic_id)
            else:
                continue
            
            entity_type = entity.get('type')
            
            try:
                if entity_type == 'LINE':
                    start = entity['start']
                    end = entity['end']
                    x_min = min(start[0], end[0])
                    y_min = min(start[1], end[1])
                    x_max = max(start[0], end[0])
                    y_max = max(start[1], end[1])
                    
                elif entity_type == 'CIRCLE':
                    center = entity.get('center', [0, 0])
                    radius = entity.get('radius', 10)
                    x_min = center[0] - radius
                    y_min = center[1] - radius
                    x_max = center[0] + radius
                    y_max = center[1] + radius
                    
                elif entity_type == 'ARC':
                    center = entity.get('center', [0, 0])
                    radius = entity.get('radius', 10)
                    x_min = center[0] - radius
                    y_min = center[1] - radius
                    x_max = center[0] + radius
                    y_max = center[1] + radius
                    
                elif entity_type == 'LWPOLYLINE':
                    # Handle polylines (take bounding box of all vertices)
                    vertices = entity.get('vertices', [])
                    if not vertices:
                        continue
                    xs = [v[0] for v in vertices]
                    ys = [v[1] for v in vertices]
                    x_min = min(xs)
                    y_min = min(ys)
                    x_max = max(xs)
                    y_max = max(ys)
                else:
                    continue
                
                # Skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Calculate normalized YOLO format
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height
                
                # Clip to valid range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))
                
                # Skip tiny boxes (noise)
                if box_width < 0.001 or box_height < 0.001:
                    continue
                
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )
            except:
                continue
        
        # Skip if no valid annotations
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

if unknown_semantics:
    print(f"\n⚠️  Unknown semantic IDs (mapped to 'wall'): {sorted(unknown_semantics)}")

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

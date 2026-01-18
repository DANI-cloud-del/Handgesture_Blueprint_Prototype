from datasets import load_dataset
from PIL import Image
import json
import os
from tqdm import tqdm
import random


# ==========================================
# FUNCTION DEFINITIONS (MUST COME FIRST!)
# ==========================================

def convert_to_yolo(json_data, image_size, class_mapping):
    """Convert ArchCAD JSON to YOLO format"""
    width, height = image_size
    yolo_annotations = []
    
    if not isinstance(json_data, list):
        return []
    
    for primitive in json_data:
        semantic = primitive.get('semantic', 'others')
        
        if semantic not in class_mapping:
            continue
        
        class_id = class_mapping[semantic]
        bbox = extract_bbox(primitive, width, height)
        
        if bbox:
            yolo_annotations.append(
                f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            )
    
    return yolo_annotations


def extract_bbox(primitive, img_width, img_height):
    """Extract normalized bounding box"""
    try:
        prim_type = primitive.get('type')
        
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
            return None
        
        # Calculate center and size
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        box_width = x_max - x_min + 10
        box_height = y_max - y_min + 10
        
        # Normalize to 0-1 range
        x_center_norm = max(0, min(1, x_center / img_width))
        y_center_norm = max(0, min(1, y_center / img_height))
        width_norm = max(0, min(1, box_width / img_width))
        height_norm = max(0, min(1, box_height / img_height))
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
        
    except:
        return None


# ==========================================
# MAIN SCRIPT STARTS HERE
# ==========================================

print("=" * 60)
print("PREPARE TRAINING DATA - STREAMING MODE")
print("=" * 60)

# Configuration
MAX_SAMPLES = 5000
VAL_SPLIT = 0.2
OUTPUT_DIR = 'archcad_yolo'

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

# Create directories
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

print(f"\n📊 Configuration:")
print(f"  - Max samples: {MAX_SAMPLES:,}")
print(f"  - Validation split: {VAL_SPLIT*100}%")
print(f"  - Output: {OUTPUT_DIR}/")
print(f"  - Classes: {len(CLASS_NAMES)}")

input("\nPress Enter to start processing...")

# Load dataset in streaming mode
print("\n🔄 Loading dataset in streaming mode...")
dataset = load_dataset(
    "jackluoluo/ArchCAD",
    split="train",
    streaming=True
)

print("✓ Dataset loaded")

train_count = 0
val_count = 0
processed = 0
skipped = 0

print(f"\n🔄 Processing {MAX_SAMPLES:,} samples...")

with tqdm(total=MAX_SAMPLES, desc="Processing") as pbar:
    for sample in dataset:
        if processed >= MAX_SAMPLES:
            break
        
        try:
            # Get image and JSON
            image = sample.get('image')
            json_data = sample.get('json')
            
            if not image or not json_data:
                skipped += 1
                continue
            
            # Parse JSON if string
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            
            # Convert to YOLO format (function defined above!)
            yolo_annotations = convert_to_yolo(json_data, image.size, CLASS_MAPPING)
            
            if not yolo_annotations:
                skipped += 1
                continue
            
            # Determine split
            is_val = random.random() < VAL_SPLIT
            split = 'val' if is_val else 'train'
            
            # Save image
            img_filename = f"archcad_{processed:06d}.jpg"
            img_path = f"{OUTPUT_DIR}/images/{split}/{img_filename}"
            
            # Resize to 640x640
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
            pbar.update(1)
            
        except Exception as e:
            skipped += 1
            continue

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

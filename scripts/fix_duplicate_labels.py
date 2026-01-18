import os
from collections import defaultdict

print("="*60)
print("FIX DUPLICATE LABELS IN DATASET")
print("="*60)

dataset_path = 'archcad_yolo'
splits = ['train', 'val']

total_duplicates = 0
files_with_duplicates = 0

for split in splits:
    labels_dir = f"{dataset_path}/labels/{split}"
    
    if not os.path.exists(labels_dir):
        print(f"⚠️  Skipping {split} (not found)")
        continue
    
    print(f"\n🔍 Processing {split} set...")
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        
        # Read all labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        duplicates_found = 0
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                unique_lines.append(line)
            elif line in seen:
                duplicates_found += 1
        
        # If duplicates found, rewrite file
        if duplicates_found > 0:
            with open(label_path, 'w') as f:
                for line in unique_lines:
                    f.write(f"{line}\n")
            
            total_duplicates += duplicates_found
            files_with_duplicates += 1
            
            if duplicates_found > 10:  # Report large numbers
                print(f"  ✓ {label_file}: {duplicates_found} duplicates removed")

print("\n" + "="*60)
print("✅ CLEANUP COMPLETE!")
print("="*60)
print(f"📊 Summary:")
print(f"  - Files with duplicates: {files_with_duplicates}")
print(f"  - Total duplicates removed: {total_duplicates}")
print(f"\n✅ Dataset is now clean!")
print(f"   Run training again: python continue_training.py")

from datasets import load_dataset

print("=" * 60)
print("TESTING ARCHCAD CONFIGURATIONS")
print("=" * 60)

# Try to find available configurations
try:
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names("jackluoluo/ArchCAD")
    print(f"\nAvailable configurations: {configs}")
except:
    print("\nCouldn't auto-detect configs, trying manual approach...")

# Test different possible config names
test_configs = ['default', 'full', 'images', 'all', 'train', None]

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing config: {config}")
    print('='*60)
    
    try:
        if config:
            dataset = load_dataset(
                "jackluoluo/ArchCAD",
                name=config,
                split="train",
                streaming=True
            )
        else:
            dataset = load_dataset(
                "jackluoluo/ArchCAD",
                split="train",
                streaming=True
            )
        
        sample = next(iter(dataset))
        
        print(f"✓ Config '{config}' works!")
        print(f"Available keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            print(f"  - {key}: {type(value).__name__}")
        
        # If we find images, stop here
        if 'image' in sample.keys():
            print("\n" + "="*60)
            print("✓✓✓ FOUND THE RIGHT CONFIG!")
            print("="*60)
            print(f"Use config name: '{config}'")
            break
            
    except Exception as e:
        print(f"✗ Config '{config}' failed: {str(e)[:100]}")

print("\n\nDone testing!")

from datasets import load_dataset
import os

print("=" * 60)
print("ARCHCAD SUBSET DOWNLOAD (OPTIMIZED FOR QUICK TRAINING)")
print("=" * 60)

# Check authentication
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"\n✓ Logged in as: {user_info['name']}")
except Exception:
    print("\n❌ Not logged in!")
    exit(1)

cache_dir = os.path.join(os.getcwd(), 'archcad_cache')
os.makedirs(cache_dir, exist_ok=True)

print(f"\n📁 Cache directory: {cache_dir}")
print("\n💡 Strategy: Download in STREAMING mode")
print("   - Process only what you need")
print("   - No 20+ hour wait!")
print("   - Perfect for 4GB GPU training\n")

input("Press Enter to continue...")

try:
    print("\n🔄 Loading dataset in streaming mode...")
    
    # Load in streaming mode (doesn't download everything at once)
    dataset = load_dataset(
        "jackluoluo/ArchCAD",
        split="train",
        streaming=True
    )
    
    print("\n✅ Dataset loaded in streaming mode!")
    print("\n📋 Sample structure:")
    
    # Get first sample to verify
    sample = next(iter(dataset))
    for key in sample.keys():
        print(f"  - {key}")
    
    print("\n" + "=" * 60)
    print("✅ READY FOR TRAINING DATA PREPARATION!")
    print("=" * 60)
    
    print("\n💡 Next step will download ONLY what's needed during processing")
    print("   Run: python scripts/prepare_training_data.py")
    print("\n   This will take ~30-60 minutes (not 20 hours!)")
    
except Exception as e:
    print(f"\n❌ Error: {e}")

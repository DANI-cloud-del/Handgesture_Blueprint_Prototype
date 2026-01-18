from datasets import load_dataset
import os

print("Testing ArchCAD download with authentication...")

# Check if logged in (updated for newer huggingface_hub)
try:
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Logged in as: {user_info['name']}")
except Exception:
    print("⚠️  Not logged in - attempting anonymous access...")

try:
    # Try loading just the dataset info (streaming mode)
    print("\nAttempting to access ArchCAD dataset...")
    dataset = load_dataset(
        "jackluoluo/ArchCAD",
        split="train",
        streaming=True  # Don't download everything, just test access
    )
    
    # Get first sample
    print("Fetching first sample...")
    sample = next(iter(dataset))
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Authentication working.")
    print("="*60)
    print(f"\n📋 Sample structure:")
    for key in sample.keys():
        print(f"  - {key}")
    print("\n✓ Ready to download full dataset!")
    
except Exception as e:
    print("\n" + "="*60)
    print("❌ ERROR")
    print("="*60)
    
    error_msg = str(e)
    print(f"\nError message: {error_msg}\n")
    
    if "gated" in error_msg.lower() or "authenticated" in error_msg.lower():
        print("Solution:")
        print("1. You're logged in, but need dataset access")
        print("2. Visit: https://huggingface.co/datasets/jackluoluo/ArchCAD")
        print("3. Click 'Agree and access repository'")
        print("4. Run this script again")
        
    elif "access" in error_msg.lower() or "denied" in error_msg.lower() or "403" in error_msg.lower():
        print("Solution:")
        print("Visit: https://huggingface.co/datasets/jackluoluo/ArchCAD")
        print("Click 'Agree and access repository'")
        
    else:
        print("Unexpected error. Details above.")

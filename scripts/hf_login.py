from huggingface_hub import login

print("=" * 60)
print("HUGGING FACE LOGIN")
print("=" * 60)

print("\n📋 Steps to get your token:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Click 'New token'")
print("3. Name: archcad")
print("4. Type: Read")
print("5. Click 'Generate token'")
print("6. Copy the token (starts with hf_...)\n")

token = input("Paste your token here: ").strip()

if not token:
    print("\n❌ No token provided!")
    exit(1)

if not token.startswith('hf_'):
    print("\n⚠️  Warning: Token should start with 'hf_'")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(1)

try:
    print("\n🔄 Logging in...")
    login(token=token)
    
    print("\n" + "=" * 60)
    print("✅ LOGIN SUCCESSFUL!")
    print("=" * 60)
    
    # Verify
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"\n✓ Logged in as: {user_info['name']}")
    print(f"✓ Token saved to: ~/.cache/huggingface/token")
    print("\n✅ You can now download the ArchCAD dataset!")
    print("\nNext step: python scripts/test_download.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ LOGIN FAILED")
    print("=" * 60)
    print(f"\nError: {e}")
    print("\nPlease check:")
    print("1. Token is correct (copy-paste from Hugging Face)")
    print("2. Token has 'read' permission")
    print("3. Internet connection is working")

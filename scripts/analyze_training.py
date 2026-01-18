import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("ANALYZE TRAINING PROGRESS")
print("="*60)

# Load training results
results_csv = 'models/archcad_detector/results.csv'

try:
    df = pd.read_csv(results_csv)
    
    # Display summary
    print("\n📊 Training Summary:")
    print(f"  Total epochs: {len(df)}")
    print(f"  Best mAP50: {df['metrics/mAP50(B)'].max():.4f}")
    print(f"  Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")
    print(f"  Final loss: {df['train/box_loss'].iloc[-1]:.4f}")
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # mAP50
    axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'])
    axes[0, 0].set_title('mAP50 (Box)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP50')
    axes[0, 0].grid(True)
    
    # Precision & Recall
    axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss
    axes[1, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    axes[1, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation Loss
    axes[1, 1].plot(df['epoch'], df['val/box_loss'], label='Box Loss')
    axes[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Class Loss')
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150)
    print("\n✓ Saved plot: training_analysis.png")
    
except FileNotFoundError:
    print(f"\n❌ Results file not found: {results_csv}")
    print("   Make sure training has completed at least one epoch")

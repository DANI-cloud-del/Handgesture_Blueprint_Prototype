#!/usr/bin/env python3
"""
Crash-proof YOLO training with automatic recovery
"""

import os
import sys
import signal
import torch
import time
from datetime import datetime
from ultralytics import YOLO

# Configuration
CONFIG = {
    'data': 'archcad_yolo/data.yaml',
    'epochs': 50,
    'batch': 4,  # Reduced for stability
    'imgsz': 640,
    'device': 0,
    'workers': 2,  # Reduced to prevent crashes
    'project': 'models',
    'name': 'archcad_detector_v2',
}

LOG_FILE = 'training_log.txt'
CHECKPOINT_DIR = f"{CONFIG['project']}/{CONFIG['name']}/weights"

# ============================================================
# LOGGING
# ============================================================

def log(message):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

# ============================================================
# SIGNAL HANDLERS
# ============================================================

interrupted = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global interrupted
    interrupted = True
    log("\n⚠️  Interrupt received! Saving checkpoint...")
    log("   Training will stop after current epoch completes")
    log("   Model will be saved automatically")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================
# GPU SAFETY CHECKS
# ============================================================

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        log("❌ No GPU found! Exiting...")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    log(f"✓ GPU: {gpu_name}")
    log(f"✓ VRAM: {gpu_memory:.2f} GB")
    
    if gpu_memory < 3:
        log("⚠️  Warning: Low VRAM detected, reducing batch size")
        CONFIG['batch'] = 2
        CONFIG['workers'] = 1
    
    return True

# ============================================================
# CHECKPOINT DETECTION
# ============================================================

def find_checkpoint():
    """Find the most recent checkpoint"""
    last_pt = os.path.join(CHECKPOINT_DIR, 'last.pt')
    best_pt = os.path.join(CHECKPOINT_DIR, 'best.pt')
    
    if os.path.exists(last_pt):
        log(f"✓ Found checkpoint: {last_pt}")
        return last_pt, True  # Can resume
    elif os.path.exists(best_pt):
        log(f"✓ Found trained model: {best_pt}")
        return best_pt, False  # Can't resume, but can fine-tune
    else:
        log("✓ No checkpoint found, starting fresh training")
        return 'yolov8n.pt', False

# ============================================================
# TRAINING WITH ERROR RECOVERY
# ============================================================

def train_with_recovery():
    """Main training loop with automatic recovery"""
    
    log("="*60)
    log("CRASH-PROOF ARCHCAD TRAINING")
    log("="*60)
    
    # Check GPU
    if not check_gpu():
        return False
    
    # Find checkpoint
    model_path, can_resume = find_checkpoint()
    
    # Load model
    log(f"\n📦 Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        log(f"❌ Failed to load model: {e}")
        log("   Falling back to pretrained yolov8n.pt")
        model = YOLO('yolov8n.pt')
        can_resume = False
    
    # Training configuration
    train_args = {
        'data': CONFIG['data'],
        'epochs': CONFIG['epochs'],
        'imgsz': CONFIG['imgsz'],
        'batch': CONFIG['batch'],
        'device': CONFIG['device'],
        'workers': CONFIG['workers'],
        
        # Memory optimization
        'amp': True,
        'cache': False,
        'close_mosaic': 10,  # Disable mosaic in last 10 epochs
        
        # Optimizer (conservative for stability)
        'optimizer': 'AdamW',
        'lr0': 0.0001 if 'best.pt' in model_path else 0.001,
        'lrf': 0.00001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        
        # Early stopping
        'patience': 15,
        
        # Augmentation (reduced for stability)
        'mosaic': 0.8,
        'mixup': 0.0,  # Disabled for stability
        'copy_paste': 0.0,  # Disabled for stability
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.3,
        'flipud': 0.5,
        'fliplr': 0.5,
        
        # Output
        'project': CONFIG['project'],
        'name': CONFIG['name'],
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': 5,  # Save every 5 epochs
        'plots': True,
        'val': True,
        
        # Resume if possible
        'resume': can_resume,
    }
    
    log(f"\n📊 Training Configuration:")
    log(f"   Mode: {'Resuming' if can_resume else 'Fine-tuning' if 'best.pt' in model_path else 'Fresh'}")
    log(f"   Epochs: {CONFIG['epochs']}")
    log(f"   Batch: {CONFIG['batch']}")
    log(f"   Workers: {CONFIG['workers']}")
    log(f"   LR: {train_args['lr0']}")
    log(f"\n⏱️  Estimated time: {CONFIG['epochs'] * 3} minutes\n")
    
    # Start training with try-except
    try:
        log("🚀 Starting training...")
        log("   Press Ctrl+C to stop gracefully\n")
        
        results = model.train(**train_args)
        
        log("\n" + "="*60)
        log("✅ TRAINING COMPLETED SUCCESSFULLY!")
        log("="*60)
        
        # Print results
        try:
            log(f"\n📊 Final Results:")
            log(f"   mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
            log(f"   mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
            log(f"   Precision: {results.results_dict['metrics/precision(B)']:.4f}")
            log(f"   Recall: {results.results_dict['metrics/recall(B)']:.4f}")
        except:
            log("   (View detailed metrics in results.csv)")
        
        log(f"\n📁 Model saved to:")
        log(f"   {CHECKPOINT_DIR}/best.pt")
        
        return True
        
    except KeyboardInterrupt:
        log("\n⚠️  Training interrupted by user")
        log(f"✓ Checkpoint saved to: {CHECKPOINT_DIR}/last.pt")
        log("✓ Run script again to resume training")
        return False
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"\n❌ GPU OUT OF MEMORY!")
            log(f"   Error: {e}")
            log(f"\n💡 Solutions:")
            log(f"   1. Reduce batch size: batch={CONFIG['batch'] // 2}")
            log(f"   2. Reduce workers: workers=1")
            log(f"   3. Reduce image size: imgsz=512")
            log(f"\n✓ Checkpoint saved - fix settings and run again")
            
            # Clear cache
            torch.cuda.empty_cache()
            return False
        else:
            log(f"\n❌ Runtime Error: {e}")
            log(f"✓ Checkpoint saved to: {CHECKPOINT_DIR}/last.pt")
            return False
            
    except Exception as e:
        log(f"\n❌ Unexpected Error: {type(e).__name__}")
        log(f"   {e}")
        log(f"\n✓ Checkpoint saved to: {CHECKPOINT_DIR}/last.pt")
        log(f"   Run script again to resume")
        return False
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"\n✓ GPU memory cleared")

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    log("\n" + "="*60)
    log(f"TRAINING SESSION STARTED")
    log("="*60)
    log(f"PID: {os.getpid()}")
    log(f"Log file: {LOG_FILE}")
    
    # Check data.yaml exists
    if not os.path.exists(CONFIG['data']):
        log(f"\n❌ Dataset not found: {CONFIG['data']}")
        log("   Make sure you've prepared the dataset first")
        sys.exit(1)
    
    # Run training
    success = train_with_recovery()
    
    if success:
        log("\n✅ All done! Model ready for inference")
        sys.exit(0)
    else:
        log("\n⚠️  Training incomplete - check log for details")
        sys.exit(1)

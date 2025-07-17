#!/usr/bin/env python3
"""
Wrapper script to run external memory training with optional metric tracking
"""

import subprocess
import sys
import os

def main():
    """Run training with or without metric tracking based on user choice"""
    print("SAPFLUXNET External Memory Training")
    print("=" * 50)
    print("Choose training mode:")
    print("1. Fast training (no metric tracking)")
    print("2. Detailed training (with metric tracking)")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    # Build command
    script_path = os.path.join(os.path.dirname(__file__), 'random_external.py')
    
    if choice == '1':
        print("\n🚀 Running fast training (no metric tracking)...")
        cmd = [sys.executable, script_path]
    else:
        print("\n📊 Running detailed training (with metric tracking)...")
        print("⚠️  This will be slower but provide detailed analysis data")
        cmd = [sys.executable, script_path, '--track-metrics']
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 
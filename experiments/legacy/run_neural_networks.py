#!/usr/bin/env python3
"""
Test script to run all neural network architectures.
This will help verify that the train.py script works with different network types.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_network_architecture(arch_name, iterations=5000):
    """Run a specific neural network architecture."""
    print(f"\n🔧 Testing architecture: {arch_name}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    cmd = f"python train.py --network {arch_name} --iterations {iterations}"
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, timeout=900, capture_output=True, text=True)  # 15 min timeout
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {arch_name} completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ {arch_name} failed after {elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {arch_name} timed out after 15 minutes")
        return False
    except Exception as e:
        print(f"💥 {arch_name} failed with exception: {e}")
        return False

def main():
    """Run all neural network architectures."""
    print("🧠 Neural Network Architecture Test")
    print("="*50)
    print("This will test all 6 network architectures with reduced iterations.")
    print("Each architecture will run for 5000 iterations (faster testing).")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('train.py'):
        print("❌ Error: train.py not found. Please run from DL_MCCFR directory.")
        sys.exit(1)
    
    architectures = [
        'simple',
        'deep_residual', 
        'feature_attention',
        'hybrid_advanced',
        'mega_transformer',
        'ultra_deep'
    ]
    
    results = {}
    
    for arch in architectures:
        success = run_network_architecture(arch, iterations=5000)
        results[arch] = success
        
        # Small delay between runs
        time.sleep(3)
    
    # Summary
    print("\n" + "="*50)
    print("📊 NEURAL NETWORK TEST RESULTS")
    print("="*50)
    
    successful = sum(results.values())
    total = len(results)
    
    for arch, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{arch:<20}: {status}")
    
    print(f"\nOverall: {successful}/{total} architectures succeeded")
    
    if successful == total:
        print("🎯 All neural network architectures working!")
    elif successful > 0:
        print(f"⚠️  {successful}/{total} architectures working. Some failed.")
    else:
        print("💥 All neural network architectures failed!")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Comprehensive script to run all possible models in the DL_MCCFR codebase.
This script will execute:
1. Robust Deep MCCFR ablation study (25+ configurations)
2. All neural network architectures (6 types)
3. Ultra deep focused experiments
4. Classical MCCFR baseline
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def run_command(cmd, description, timeout=None):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description} completed in {elapsed:.1f}s")
            return True, result.stdout
        else:
            print(f"❌ FAILED: {description} failed after {elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} timed out after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 ERROR: {description} failed with exception: {e}")
        return False, str(e)

def run_robust_deep_mccfr_ablation():
    """Run the complete robust deep MCCFR ablation study."""
    print("\n🚀 Starting Robust Deep MCCFR Ablation Study")
    print("This will run 25+ different model configurations...")
    
    cmd = "python deep_mccfr.py --experiment ablation --iterations 10000"
    success, output = run_command(cmd, "Robust Deep MCCFR Ablation Study", timeout=3600)  # 1 hour timeout
    
    if success:
        print("✅ Ablation study completed successfully!")
        # Try to find the results directory
        for item in os.listdir('.'):
            if item.startswith('robust_mccfr_experiments_'):
                print(f"📁 Results saved in: {item}")
                break
    else:
        print("❌ Ablation study failed")
    
    return success

def run_neural_network_architectures():
    """Run all neural network architecture types."""
    print("\n🧠 Starting Neural Network Architecture Experiments")
    
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
        print(f"\n🔧 Testing architecture: {arch}")
        cmd = f"python train.py --network {arch} --iterations 10000"
        success, output = run_command(cmd, f"Neural Network: {arch}", timeout=1800)  # 30 min timeout
        
        results[arch] = {
            'success': success,
            'output': output
        }
        
        if success:
            print(f"✅ {arch} architecture completed successfully!")
        else:
            print(f"❌ {arch} architecture failed")
        
        # Small delay between runs
        time.sleep(5)
    
    return results

def run_ultra_deep_focused_experiments():
    """Run ultra deep focused experiments."""
    print("\n⚡ Starting Ultra Deep Focused Experiments")
    
    cmd = "python focused_ultra_deep_experiments.py --mode focused --iterations 10000"
    success, output = run_command(cmd, "Ultra Deep Focused Experiments", timeout=3600)  # 1 hour timeout
    
    if success:
        print("✅ Ultra deep focused experiments completed successfully!")
    else:
        print("❌ Ultra deep focused experiments failed")
    
    return success

def run_classical_mccfr():
    """Run classical MCCFR baseline."""
    print("\n🎯 Starting Classical MCCFR Baseline")
    
    cmd = "python train_classical.py --iterations 10000"
    success, output = run_command(cmd, "Classical MCCFR", timeout=1800)  # 30 min timeout
    
    if success:
        print("✅ Classical MCCFR completed successfully!")
    else:
        print("❌ Classical MCCFR failed")
    
    return success

def run_improved_train():
    """Run improved training experiments."""
    print("\n🚀 Starting Improved Training Experiments")
    
    cmd = "python improved_train.py --iterations 10000"
    success, output = run_command(cmd, "Improved Training", timeout=1800)  # 30 min timeout
    
    if success:
        print("✅ Improved training completed successfully!")
    else:
        print("❌ Improved training failed")
    
    return success

def generate_summary_report(results):
    """Generate a summary report of all experiments."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"all_models_summary_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("ALL MODELS EXECUTION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENT RESULTS:\n")
        f.write("-"*30 + "\n")
        
        for experiment, result in results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            f.write(f"{experiment}: {status}\n")
        
        f.write(f"\nDetailed results saved in individual experiment directories.\n")
    
    print(f"\n📊 Summary report generated: {report_file}")
    return report_file

def main():
    """Main execution function."""
    print("🌟 DL_MCCFR - All Models Execution Script")
    print("="*60)
    print("This script will run ALL available models and experiments.")
    print("Estimated total time: 3-4 hours")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('deep_mccfr.py'):
        print("❌ Error: Please run this script from the DL_MCCFR directory")
        sys.exit(1)
    
    # Store all results
    all_results = {}
    
    # 1. Run Robust Deep MCCFR Ablation Study
    print("\n" + "="*80)
    print("PHASE 1: ROBUST DEEP MCCFR ABLATION STUDY")
    print("="*80)
    all_results['robust_deep_mccfr_ablation'] = {
        'success': run_robust_deep_mccfr_ablation(),
        'output': 'See individual experiment directories'
    }
    
    # 2. Run Neural Network Architectures
    print("\n" + "="*80)
    print("PHASE 2: NEURAL NETWORK ARCHITECTURES")
    print("="*80)
    all_results['neural_network_architectures'] = run_neural_network_architectures()
    
    # 3. Run Ultra Deep Focused Experiments
    print("\n" + "="*80)
    print("PHASE 3: ULTRA DEEP FOCUSED EXPERIMENTS")
    print("="*80)
    all_results['ultra_deep_focused'] = {
        'success': run_ultra_deep_focused_experiments(),
        'output': 'See individual experiment directories'
    }
    
    # 4. Run Classical MCCFR
    print("\n" + "="*80)
    print("PHASE 4: CLASSICAL MCCFR BASELINE")
    print("="*80)
    all_results['classical_mccfr'] = {
        'success': run_classical_mccfr(),
        'output': 'See individual experiment directories'
    }
    
    # 5. Run Improved Training
    print("\n" + "="*80)
    print("PHASE 5: IMPROVED TRAINING EXPERIMENTS")
    print("="*80)
    all_results['improved_training'] = {
        'success': run_improved_train(),
        'output': 'See individual experiment directories'
    }
    
    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    report_file = generate_summary_report(all_results)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ALL MODELS EXECUTION COMPLETED!")
    print("="*80)
    
    successful = sum(1 for r in all_results.values() 
                    if isinstance(r, dict) and r.get('success', False))
    total = len(all_results)
    
    print(f"📊 Results Summary:")
    print(f"   ✅ Successful: {successful}/{total}")
    print(f"   ❌ Failed: {total - successful}/{total}")
    print(f"📁 Summary report: {report_file}")
    print(f"🔍 Check individual experiment directories for detailed results")
    
    if successful == total:
        print("🎯 All models executed successfully!")
    else:
        print("⚠️  Some models failed. Check the summary report for details.")

if __name__ == "__main__":
    main()



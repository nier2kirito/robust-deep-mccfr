#!/usr/bin/env python3
"""
Run the complete Robust Deep MCCFR ablation study.
This will execute 25+ different model configurations to systematically evaluate
the impact of each component in the robust deep MCCFR framework.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def main():
    """Run the complete ablation study."""
    print("🚀 Robust Deep MCCFR - Complete Ablation Study")
    print("="*60)
    print("This will run 25+ different model configurations including:")
    print("• Baseline configuration")
    print("• Component ablations (no_exploration_mixing, no_weight_clipping, etc.)")
    print("• Parameter variations (different epsilon values, weight clipping, etc.)")
    print("• Combination studies (minimal_risk, maximal_risk, balanced_approach)")
    print("="*60)
    print("Estimated time: 1-2 hours")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('deep_mccfr.py'):
        print("❌ Error: Please run this script from the DL_MCCFR directory")
        sys.exit(1)
    
    # Run the ablation study
    cmd = "python deep_mccfr.py --experiment ablation --iterations 10000"
    
    print(f"\n🔧 Executing command: {cmd}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results will be saved in a timestamped directory")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(cmd, shell=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Ablation study completed successfully!")
        print(f"⏱️  Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Find the results directory
        for item in os.listdir('.'):
            if item.startswith('robust_mccfr_experiments_'):
                print(f"📁 Results saved in: {item}")
                
                # Check for analysis file
                analysis_file = os.path.join(item, "ablation_analysis.txt")
                if os.path.exists(analysis_file):
                    print(f"📊 Analysis report: {analysis_file}")
                    
                    # Show a preview of the results
                    print("\n📈 Top Results Preview:")
                    try:
                        with open(analysis_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines[:10]:  # Show first 10 lines
                                print(f"   {line.rstrip()}")
                    except:
                        pass
                break
                
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ablation study failed with error code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Ablation study interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



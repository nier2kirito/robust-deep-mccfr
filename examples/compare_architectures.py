#!/usr/bin/env python3
"""
Architecture Comparison Example

This script compares different neural network architectures for Deep MCCFR
and generates a comprehensive comparison report.
"""

import sys
import os
import argparse
import time
import json
from typing import Dict, List

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dl_mccfr import DeepMCCFR
from dl_mccfr.networks import count_parameters


def run_architecture_comparison(architectures: List[str], iterations: int = 5000) -> Dict:
    """
    Run training with different architectures and compare results.
    
    Args:
        architectures: List of architecture names to compare
        iterations: Number of training iterations for each architecture
        
    Returns:
        Dictionary containing comparison results
    """
    results = {}
    
    print("="*80)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("="*80)
    print(f"Comparing {len(architectures)} architectures with {iterations} iterations each")
    print("-"*80)
    
    for i, arch in enumerate(architectures, 1):
        print(f"\n[{i}/{len(architectures)}] Training with {arch} architecture...")
        
        try:
            # Initialize MCCFR with current architecture
            mccfr = DeepMCCFR(
                network_type=arch,
                learning_rate=0.00003,
                batch_size=256,  # Smaller batch for fair comparison
                train_every=25
            )
            
            # Count parameters
            policy_params = count_parameters(mccfr.policy_net)
            sampler_params = count_parameters(mccfr.sampler_net)
            total_params = policy_params + sampler_params
            
            print(f"  Parameters: {total_params:,} ({policy_params:,} + {sampler_params:,})")
            
            # Train the model
            start_time = time.time()
            train_results = mccfr.train(num_iterations=iterations)
            training_time = time.time() - start_time
            
            # Store results
            results[arch] = {
                'parameters': {
                    'policy': policy_params,
                    'sampler': sampler_params,
                    'total': total_params
                },
                'performance': {
                    'final_exploitability': train_results['final_exploitability'],
                    'training_time': training_time,
                    'total_infosets': train_results['total_infosets']
                },
                'training_metrics': train_results['training_metrics']
            }
            
            print(f"  Final Exploitability: {train_results['final_exploitability']:.6f}")
            print(f"  Training Time: {training_time:.1f}s")
            print(f"  Status: ✓ Completed successfully")
            
        except Exception as e:
            print(f"  Status: ✗ Failed with error: {e}")
            results[arch] = {
                'error': str(e),
                'status': 'failed'
            }
    
    return results


def analyze_results(results: Dict) -> None:
    """
    Analyze and print comparison results.
    
    Args:
        results: Results dictionary from run_architecture_comparison
    """
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if 'error' not in v and v.get('performance', {}).get('final_exploitability') is not None}
    
    if not successful_results:
        print("No successful training runs to analyze.")
        return
    
    # Sort by exploitability (lower is better)
    sorted_results = sorted(successful_results.items(), 
                          key=lambda x: x[1]['performance']['final_exploitability'])
    
    print("\n1. PERFORMANCE RANKING (by final exploitability)")
    print("-" * 50)
    print(f"{'Rank':<4} {'Architecture':<20} {'Exploitability':<15} {'Time (s)':<10} {'Parameters':<12}")
    print("-" * 50)
    
    for rank, (arch, data) in enumerate(sorted_results, 1):
        exploit = data['performance']['final_exploitability']
        time_taken = data['performance']['training_time']
        params = data['parameters']['total']
        
        print(f"{rank:<4} {arch:<20} {exploit:<15.6f} {time_taken:<10.1f} {params:<12,}")
    
    # Parameter efficiency analysis
    print("\n2. PARAMETER EFFICIENCY")
    print("-" * 50)
    print(f"{'Architecture':<20} {'Params (M)':<12} {'Exploit/Param':<15} {'Efficiency':<10}")
    print("-" * 50)
    
    for arch, data in sorted_results:
        params_m = data['parameters']['total'] / 1e6
        exploit = data['performance']['final_exploitability']
        efficiency = exploit / params_m if params_m > 0 else float('inf')
        
        # Efficiency score (lower is better - less exploitability per parameter)
        print(f"{arch:<20} {params_m:<12.2f} {efficiency:<15.6f} {'High' if efficiency < 0.001 else 'Medium' if efficiency < 0.01 else 'Low':<10}")
    
    # Training time analysis
    print("\n3. TRAINING TIME ANALYSIS")
    print("-" * 50)
    print(f"{'Architecture':<20} {'Time (s)':<12} {'Time/Param (μs)':<15} {'Speed':<10}")
    print("-" * 50)
    
    for arch, data in sorted_results:
        time_taken = data['performance']['training_time']
        params = data['parameters']['total']
        time_per_param = (time_taken / params) * 1e6 if params > 0 else 0
        
        speed = 'Fast' if time_per_param < 0.1 else 'Medium' if time_per_param < 1.0 else 'Slow'
        print(f"{arch:<20} {time_taken:<12.1f} {time_per_param:<15.2f} {speed:<10}")
    
    # Best performer analysis
    best_arch, best_data = sorted_results[0]
    print(f"\n4. BEST PERFORMER: {best_arch}")
    print("-" * 50)
    print(f"Final Exploitability: {best_data['performance']['final_exploitability']:.6f}")
    print(f"Training Time: {best_data['performance']['training_time']:.1f}s")
    print(f"Total Parameters: {best_data['parameters']['total']:,}")
    print(f"Information Sets: {best_data['performance']['total_infosets']}")
    
    # Learning curve analysis
    if best_data['training_metrics']['exploitability']:
        exploit_history = best_data['training_metrics']['exploitability']
        initial_exploit = exploit_history[0]
        final_exploit = exploit_history[-1]
        improvement = (initial_exploit - final_exploit) / initial_exploit * 100
        
        print(f"Initial Exploitability: {initial_exploit:.6f}")
        print(f"Improvement: {improvement:.2f}%")
        print(f"Best Exploitability: {min(exploit_history):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Compare Neural Network Architectures')
    parser.add_argument('--architectures', nargs='+', 
                       default=['simple', 'deep_residual', 'feature_attention', 'ultra_deep'],
                       choices=['simple', 'deep_residual', 'feature_attention', 
                               'hybrid_advanced', 'mega_transformer', 'ultra_deep'],
                       help='Architectures to compare')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of training iterations per architecture')
    parser.add_argument('--output', type=str, default='architecture_comparison.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_architecture_comparison(args.architectures, args.iterations)
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    def make_json_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_json_serializable(results)
    serializable_results['config'] = {
        'architectures': args.architectures,
        'iterations': args.iterations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(args.output, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Comparison completed! Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()

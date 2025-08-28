#!/usr/bin/env python3
"""
Basic Deep MCCFR Training Example

This script demonstrates how to use the DL-MCCFR library for basic training
on Kuhn Poker with different neural network architectures.
"""

import sys
import os
import argparse
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dl_mccfr import DeepMCCFR, KuhnGame
from dl_mccfr.networks import count_parameters


def main():
    parser = argparse.ArgumentParser(description='Basic Deep MCCFR Training')
    parser.add_argument('--network', type=str, default='ultra_deep',
                       choices=['simple', 'deep_residual', 'feature_attention', 
                               'hybrid_advanced', 'mega_transformer', 'ultra_deep'],
                       help='Neural network architecture to use')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of training iterations')
    parser.add_argument('--learning-rate', type=float, default=0.00003,
                       help='Learning rate for optimization')
    parser.add_argument('--batch-size', type=int, default=384,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        device = torch.device(args.device)
    
    print("="*60)
    print("Deep MCCFR Basic Training Example")
    print("="*60)
    print(f"Network Architecture: {args.network}")
    print(f"Training Iterations: {args.iterations}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print("-"*60)
    
    # Initialize Deep MCCFR
    mccfr = DeepMCCFR(
        network_type=args.network,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=device
    )
    
    print(f"Policy Network Parameters: {count_parameters(mccfr.policy_net):,}")
    print(f"Sampler Network Parameters: {count_parameters(mccfr.sampler_net):,}")
    print(f"Total Parameters: {count_parameters(mccfr.policy_net) + count_parameters(mccfr.sampler_net):,}")
    print("-"*60)
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    results = mccfr.train(num_iterations=args.iterations)
    
    training_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Final Exploitability: {results['final_exploitability']:.6f}")
    print(f"Training Time: {results['training_time']:.1f}s")
    print(f"Total Information Sets: {results['total_infosets']}")
    
    # Print training metrics summary
    if results['training_metrics']['exploitability']:
        exploitability_history = results['training_metrics']['exploitability']
        print(f"Initial Exploitability: {exploitability_history[0]:.6f}")
        print(f"Best Exploitability: {min(exploitability_history):.6f}")
        improvement = (exploitability_history[0] - exploitability_history[-1]) / exploitability_history[0] * 100
        print(f"Improvement: {improvement:.2f}%")
    
    if results['training_metrics']['policy_losses']:
        policy_losses = results['training_metrics']['policy_losses']
        print(f"Initial Policy Loss: {policy_losses[0]:.6f}")
        print(f"Final Policy Loss: {policy_losses[-1]:.6f}")
    
    print("-"*60)
    print("Training completed successfully!")
    
    # Optionally save results
    import json
    output_file = f"results_{args.network}_{args.iterations}iters.json"
    
    # Convert numpy arrays to lists for JSON serialization
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
        'network_type': args.network,
        'iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'device': str(device)
    }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comparison script for different neural network architectures in MCCFR training.
This script shows the parameter counts and provides quick training runs to compare performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

# GPU/Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import the architectures from train2.py (assuming it's in the same directory)
try:
    from train2 import (
        BaseNN, DeepResidualNN, FeatureAttentionNN, HybridAdvancedNN, 
        MegaTransformerNN, UltraDeepNN,
        NETWORK_CONFIGS, create_network, count_parameters,
        INPUT_SIZE, NUM_TOTAL_ACTIONS
    )
except ImportError:
    print("Error: Could not import from train2.py. Make sure it's in the same directory.")
    exit(1)

def benchmark_architecture(network_type, num_forward_passes=1000):
    """Benchmark the forward pass speed of a network architecture."""
    print(f"\n--- Benchmarking {network_type} Architecture ---")
    
    # Create network and move to device
    config = NETWORK_CONFIGS[network_type]
    network = config['class'](
        input_size=INPUT_SIZE,
        hidden_size=config['hidden_size'],
        num_actions=NUM_TOTAL_ACTIONS,
        **config['kwargs']
    ).to(device)
    
    # Count parameters
    param_count = count_parameters(network)
    print(f"Parameters: {param_count:,}")
    
    # Create dummy input and move to device
    dummy_input = torch.randn(32, INPUT_SIZE, device=device)  # Batch size of 32
    
    # Warm up
    network.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = network(dummy_input)
    
    # Synchronize GPU for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark forward passes
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_forward_passes):
            output = network(dummy_input)
    
    # Synchronize GPU for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_pass = total_time / num_forward_passes * 1000  # Convert to milliseconds
    
    print(f"Forward pass time: {time_per_pass:.3f} ms per batch (batch size 32)")
    print(f"Total benchmark time: {total_time:.2f} seconds")
    
    # Test output shape and properties
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be close to batch_size={dummy_input.shape[0]}): {torch.sum(output):.6f}")
    print(f"Min/Max output values: {torch.min(output):.6f} / {torch.max(output):.6f}")
    
    return param_count, time_per_pass

def compare_all_architectures():
    """Compare all available architectures."""
    print("=" * 80)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    results = {}
    
    for arch_name in ['simple', 'deep_residual', 'feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
        try:
            param_count, time_per_pass = benchmark_architecture(arch_name)
            results[arch_name] = {
                'params': param_count,
                'time': time_per_pass
            }
        except Exception as e:
            print(f"Error testing {arch_name}: {e}")
            results[arch_name] = {
                'params': 0,
                'time': float('inf')
            }
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Architecture':<20} {'Parameters':<15} {'Time (ms)':<12} {'Params/Time Ratio':<15}")
    print("-" * 80)
    
    for arch_name, metrics in results.items():
        if metrics['time'] != float('inf') and metrics['time'] > 0:
            efficiency = metrics['params'] / metrics['time']
            print(f"{arch_name:<20} {metrics['params']:<15,} {metrics['time']:<12.3f} {efficiency:<15.1f}")
        else:
            print(f"{arch_name:<20} {metrics['params']:<15,} {'Error':<12} {'N/A':<15}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best architectures
    valid_results = {k: v for k, v in results.items() if v['time'] != float('inf')}
    
    if valid_results:
        # Most parameters (highest capacity)
        max_params_arch = max(valid_results.keys(), key=lambda x: valid_results[x]['params'])
        print(f"🧠 Highest Capacity: {max_params_arch} ({valid_results[max_params_arch]['params']:,} parameters)")
        
        # Fastest
        fastest_arch = min(valid_results.keys(), key=lambda x: valid_results[x]['time'])
        print(f"⚡ Fastest: {fastest_arch} ({valid_results[fastest_arch]['time']:.3f} ms per batch)")
        
        # Best efficiency (params per unit time)
        if all(v['time'] > 0 for v in valid_results.values()):
            most_efficient_arch = max(valid_results.keys(), 
                                    key=lambda x: valid_results[x]['params'] / valid_results[x]['time'])
            efficiency = valid_results[most_efficient_arch]['params'] / valid_results[most_efficient_arch]['time']
            print(f"📊 Most Efficient: {most_efficient_arch} ({efficiency:.1f} params/ms)")
    
    print("\n💡 Usage suggestions:")
    print("  • For quick experiments: use 'simple'")
    print("  • For better performance with moderate complexity: use 'deep_residual'")  
    print("  • For maximum expressiveness: use 'feature_attention' or 'hybrid_advanced'")
    print("  • For production with limited compute: consider speed vs accuracy trade-off")

def test_training_step():
    """Test a single training step with different loss functions."""
    print("\n" + "=" * 80)
    print("TESTING TRAINING STEP")
    print("=" * 80)
    
    # Create dummy training data and move to device
    batch_size = 64
    features = torch.randn(batch_size, INPUT_SIZE, device=device)
    targets = torch.softmax(torch.randn(batch_size, NUM_TOTAL_ACTIONS, device=device), dim=1)
    
    for arch_name in ['simple', 'mega_transformer']:
        print(f"\n--- Testing {arch_name} training step ---")
        
        # Create network and optimizer, move to device
        network = create_network(arch_name).to(device)
        optimizer = optim.AdamW(network.parameters(), lr=0.001)
        
        # Training step
        network.train()
        optimizer.zero_grad()
        
        predictions = network(features)
        
        # Test different loss functions
        mse_loss = nn.MSELoss()(predictions, targets)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(predictions + 1e-9), targets)
        
        print(f"MSE Loss: {mse_loss.item():.6f}")
        print(f"KL Divergence Loss: {kl_loss.item():.6f}")
        
        # Test backpropagation
        try:
            mse_loss.backward()
            # Check if gradients exist
            total_grad_norm = sum(p.grad.norm().item() for p in network.parameters() if p.grad is not None)
            print(f"Total gradient norm: {total_grad_norm:.6f}")
            print("✅ Backpropagation successful")
        except Exception as e:
            print(f"❌ Backpropagation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare neural network architectures for MCCFR')
    parser.add_argument('--benchmark-passes', type=int, default=1000,
                       help='Number of forward passes for benchmarking (default: 1000)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step tests')
    
    args = parser.parse_args()
    
    # Run comparison
    compare_all_architectures()
    
    if not args.skip_training:
        test_training_step()
    
    print(f"\n🎯 To use a specific architecture in training, run:")
    print(f"   python train2.py --network <architecture_name> --iterations <num_iterations>")
    print(f"\nAvailable architectures: simple, deep_residual, feature_attention, hybrid_advanced, mega_transformer, ultra_deep") 
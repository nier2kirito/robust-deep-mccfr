import json
import os
import numpy as np
import argparse
from datetime import datetime
import glob

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Only text analysis will be provided.")

def load_experiment_results(experiment_dir):
    """Load all experiment results from a directory."""
    
    summary_file = os.path.join(experiment_dir, "experiment_summary.json")
    
    # Try to load from summary first
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Check if the summary has config data
        results = summary_data['results']
        if results and 'config' in results[0] and isinstance(results[0]['config'], dict):
            print(f"Loaded from summary file: {len(results)} experiments")
            return results
        else:
            print("Summary file missing config data, loading from individual files...")
    
    # Fallback: load individual result files
    results = []
    result_files = glob.glob(os.path.join(experiment_dir, "*_result.json"))
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    print(f"Loaded from individual files: {len(results)} experiments")
    return results

def analyze_parameter_impact(results, parameter_name):
    """Analyze the impact of a specific parameter on performance."""
    
    param_groups = {}
    
    for result in results:
        if result.get('final_exploitability') is None:
            continue
        
        # Check if config exists and has the parameter
        config = result.get('config', {})
        if not isinstance(config, dict) or parameter_name not in config:
            continue
            
        param_value = config[parameter_name]
        
        if param_value not in param_groups:
            param_groups[param_value] = []
        
        param_groups[param_value].append(result['final_exploitability'])
    
    # Calculate statistics for each parameter value
    param_stats = {}
    for value, exploitabilities in param_groups.items():
        param_stats[value] = {
            'mean': np.mean(exploitabilities),
            'std': np.std(exploitabilities),
            'min': np.min(exploitabilities),
            'max': np.max(exploitabilities),
            'count': len(exploitabilities)
        }
    
    return param_stats

def print_parameter_analysis(results):
    """Print detailed parameter analysis."""
    
    print("\n" + "="*80)
    print("DETAILED PARAMETER ANALYSIS")
    print("="*80)
    
    key_params = [
        'warm_start_min_visits', 
        'num_blocks', 
        'bottleneck_factor',
        'hidden_size', 
        'learning_rate', 
        'weight_decay',
        'batch_size', 
        'train_every',
        'training_data_collection_threshold'
    ]
    
    for param in key_params:
        print(f"\n{param.upper().replace('_', ' ')} ANALYSIS:")
        print("-" * 50)
        
        param_stats = analyze_parameter_impact(results, param)
        
        if not param_stats:
            print("  No data available")
            continue
        
        # Sort by mean performance (lower is better)
        sorted_stats = sorted(param_stats.items(), key=lambda x: x[1]['mean'])
        
        print(f"{'Value':<15} {'Mean':<10} {'Std':<8} {'Min':<8} {'Max':<8} {'Count':<6}")
        print("-" * 65)
        
        for value, stats in sorted_stats:
            print(f"{value:<15} {stats['mean']:<10.4f} {stats['std']:<8.4f} "
                  f"{stats['min']:<8.4f} {stats['max']:<8.4f} {stats['count']:<6}")

def print_top_performers(results, top_n=10):
    """Print the top performing experiments."""
    
    # Filter out failed experiments
    valid_results = [r for r in results if r.get('final_exploitability') is not None]
    
    if not valid_results:
        print("No valid results to analyze.")
        return
    
    # Sort by final exploitability (lower is better)
    valid_results.sort(key=lambda x: x['final_exploitability'])
    
    print(f"\nTOP {top_n} PERFORMING EXPERIMENTS:")
    print("="*100)
    print(f"{'Rank':<4} {'Experiment ID':<25} {'Final Exploit.':<12} {'Best Exploit.':<12} {'Params':<12} {'Time(s)':<8}")
    print("-" * 100)
    
    for i, result in enumerate(valid_results[:top_n]):
        params_millions = result['total_parameters'] / 1_000_000
        experiment_id = result.get('experiment_id', 'unknown')
        final_exploit = result.get('final_exploitability', 0)
        best_exploit = result.get('best_exploitability', 0)
        training_time = result.get('training_time_seconds', 0)
        
        print(f"{i+1:<4} {experiment_id:<25} "
              f"{final_exploit:<12.6f} "
              f"{best_exploit:<12.6f} "
              f"{params_millions:<12.1f}M "
              f"{training_time:<8.1f}")
    
    print(f"\nBEST EXPERIMENT DETAILS:")
    print("-" * 50)
    best = valid_results[0]
    print(f"Experiment ID: {best.get('experiment_id', 'unknown')}")
    print(f"Final Exploitability: {best.get('final_exploitability', 0):.6f}")
    print(f"Best Exploitability: {best.get('best_exploitability', 0):.6f}")
    print(f"Total Parameters: {best.get('total_parameters', 0):,}")
    print(f"Training Time: {best.get('training_time_seconds', 0):.1f} seconds")
    
    # Check if config exists and handle accordingly
    config = best.get('config', {})
    if isinstance(config, dict):
        print(f"\nBest Configuration:")
        key_config_params = [
            'warm_start_min_visits', 'num_blocks', 'bottleneck_factor', 'hidden_size',
            'learning_rate', 'weight_decay', 'batch_size', 'train_every'
        ]
        
        for param in key_config_params:
            if param in config:
                print(f"  {param}: {config[param]}")
    else:
        print(f"\nConfiguration data not available or in unexpected format.")

def plot_parameter_trends(results):
    """Create plots showing parameter trends."""
    
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Install matplotlib and seaborn for visualizations.")
        return
    
    # Filter valid results
    valid_results = [r for r in results if r['final_exploitability'] is not None]
    
    if len(valid_results) < 3:
        print("Not enough valid results for plotting.")
        return
    
    # Create plots for key parameters
    key_params = ['warm_start_min_visits', 'num_blocks', 'learning_rate', 'bottleneck_factor']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, param in enumerate(key_params):
        if i >= len(axes):
            break
            
        # Extract data for this parameter
        param_values = []
        exploitabilities = []
        
        for result in valid_results:
            param_values.append(result['config'][param])
            exploitabilities.append(result['final_exploitability'])
        
        # Create scatter plot
        axes[i].scatter(param_values, exploitabilities, alpha=0.7)
        axes[i].set_xlabel(param.replace('_', ' ').title())
        axes[i].set_ylabel('Final Exploitability')
        axes[i].set_title(f'Impact of {param.replace("_", " ").title()}')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line if we have enough points
        if len(set(param_values)) > 2:
            z = np.polyfit(param_values, exploitabilities, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(param_values), max(param_values), 100)
            axes[i].plot(x_line, p(x_line), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("\nParameter analysis plot saved as 'parameter_analysis.png'")
    plt.show()

def compare_experiment_dirs(dir1, dir2):
    """Compare results from two different experiment directories."""
    
    print(f"\nCOMPARING EXPERIMENT DIRECTORIES:")
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print("="*80)
    
    results1 = load_experiment_results(dir1)
    results2 = load_experiment_results(dir2)
    
    # Get valid results
    valid1 = [r for r in results1 if r['final_exploitability'] is not None]
    valid2 = [r for r in results2 if r['final_exploitability'] is not None]
    
    if not valid1 or not valid2:
        print("One or both directories have no valid results.")
        return
    
    # Find best from each
    best1 = min(valid1, key=lambda x: x['final_exploitability'])
    best2 = min(valid2, key=lambda x: x['final_exploitability'])
    
    print(f"\nBest from Directory 1:")
    print(f"  ID: {best1['experiment_id']}")
    print(f"  Final Exploitability: {best1['final_exploitability']:.6f}")
    
    print(f"\nBest from Directory 2:")
    print(f"  ID: {best2['experiment_id']}")
    print(f"  Final Exploitability: {best2['final_exploitability']:.6f}")
    
    improvement = ((best1['final_exploitability'] - best2['final_exploitability']) / 
                   best1['final_exploitability'] * 100)
    
    if improvement > 0:
        print(f"\nDirectory 2 is {improvement:.2f}% better than Directory 1")
    else:
        print(f"\nDirectory 1 is {-improvement:.2f}% better than Directory 2")

def main():
    parser = argparse.ArgumentParser(description='Analyze Ultra Deep Parameter Experiment Results')
    parser.add_argument('experiment_dir', help='Directory containing experiment results')
    parser.add_argument('--compare', help='Second directory to compare with')
    parser.add_argument('--plot', action='store_true', help='Generate plots (requires matplotlib)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top performers to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Directory {args.experiment_dir} does not exist.")
        return
    
    print(f"Loading experiment results from: {args.experiment_dir}")
    results = load_experiment_results(args.experiment_dir)
    
    if not results:
        print("No experiment results found.")
        return
    
    print(f"Loaded {len(results)} experiment results.")
    
    # Print top performers
    print_top_performers(results, args.top_n)
    
    # Print detailed parameter analysis
    print_parameter_analysis(results)
    
    # Generate plots if requested
    if args.plot:
        plot_parameter_trends(results)
    
    # Compare with another directory if specified
    if args.compare:
        compare_experiment_dirs(args.experiment_dir, args.compare)

if __name__ == "__main__":
    main() 
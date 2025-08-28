import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
import time
import argparse
import json
import os
from datetime import datetime
import itertools

# GPU/Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import the existing components
from games.kuhn import KuhnGame, KuhnState, Action, Card, card_to_string
from utils import KuhnStrategy, calculate_exploitability
from train2 import (
    get_state_features, UltraDeepNN, count_parameters, 
    convert_mccfr_to_kuhn_strategies, train_neural_networks_on_batch,
    INPUT_SIZE, NUM_TOTAL_ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION
)

# Import the base experiment infrastructure
from ultra_deep_parameter_experiments import ExperimentConfig, ExperimentRunner

def get_focused_experiment_configs():
    """Define focused experiment configurations based on initial findings."""
    
    configs = []
    
    # Based on initial results, these parameters showed promise:
    # - warm_start_min_visits: 75, 150 performed best
    # - num_blocks: 25 performed well
    # - learning_rate: 3e-05 performed better than 5e-05
    # - batch_size: 384 performed better than 512
    
    # 1. Explore warm_start_min_visits around the optimal values (75, 150)
    promising_visits = [60, 75, 90, 120, 150, 180]
    for visits in promising_visits:
        configs.append(ExperimentConfig(
            warm_start_min_visits=visits,
            name=f"visits_{visits}_focused"
        ))
    
    # 2. Explore num_blocks around the optimal value (25)
    promising_blocks = [20, 22, 25, 28, 30]
    for blocks in promising_blocks:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,  # Use best visits value
            num_blocks=blocks,
            name=f"blocks_{blocks}_focused"
        ))
    
    # 3. Explore bottleneck_factor (original was 4)
    promising_bottleneck = [2, 3, 4, 5, 6]
    for bottleneck in promising_bottleneck:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=25,
            bottleneck_factor=bottleneck,
            name=f"bottleneck_{bottleneck}_focused"
        ))
    
    # 4. Explore learning rates around the optimal value
    promising_lrs = [0.000015, 0.00002, 0.00003, 0.000035, 0.00004]
    for lr in promising_lrs:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=25,
            learning_rate=lr,
            name=f"lr_{lr}_focused"
        ))
    
    # 5. Explore hidden sizes
    promising_hidden = [1280, 1536, 1792, 2048]
    for hidden in promising_hidden:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=25,
            hidden_size=hidden,
            name=f"hidden_{hidden}_focused"
        ))
    
    # 6. Explore weight_decay
    promising_wd = [0.0001, 0.0003, 0.0005, 0.001]
    for wd in promising_wd:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=25,
            weight_decay=wd,
            name=f"wd_{wd}_focused"
        ))
    
    # 7. Explore training_data_collection_threshold
    promising_thresholds = [10, 15, 20, 25, 30]
    for threshold in promising_thresholds:
        configs.append(ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=25,
            training_data_collection_threshold=threshold,
            name=f"threshold_{threshold}_focused"
        ))
    
    # 8. Best combinations based on initial findings
    best_combinations = [
        # Ultra-optimized configuration
        ExperimentConfig(
            warm_start_min_visits=75,
            training_data_collection_threshold=15,
            num_blocks=25,
            bottleneck_factor=3,
            hidden_size=1792,
            learning_rate=0.00003,
            weight_decay=0.0003,
            batch_size=384,
            train_every=25,
            gradient_clip_norm=0.8,
            name="ultra_optimized"
        ),
        
        # Fast convergence optimized
        ExperimentConfig(
            warm_start_min_visits=60,
            training_data_collection_threshold=10,
            num_blocks=22,
            bottleneck_factor=4,
            hidden_size=1536,
            learning_rate=0.000035,
            weight_decay=0.0001,
            batch_size=320,
            train_every=20,
            name="fast_convergence_optimized"
        ),
        
        # High capacity optimized
        ExperimentConfig(
            warm_start_min_visits=90,
            training_data_collection_threshold=20,
            num_blocks=30,
            bottleneck_factor=2,
            hidden_size=2048,
            learning_rate=0.00002,
            weight_decay=0.0005,
            batch_size=384,
            train_every=30,
            name="high_capacity_optimized"
        ),
        
        # Stable training optimized
        ExperimentConfig(
            warm_start_min_visits=120,
            training_data_collection_threshold=25,
            num_blocks=25,
            bottleneck_factor=4,
            hidden_size=1536,
            learning_rate=0.00003,
            weight_decay=0.001,
            batch_size=384,
            train_every=25,
            gradient_clip_norm=0.5,
            warmup_ratio=0.15,
            name="stable_training_optimized"
        ),
        
        # Exploration-focused
        ExperimentConfig(
            warm_start_min_visits=75,
            training_data_collection_threshold=10,
            num_blocks=25,
            bottleneck_factor=3,
            hidden_size=1792,
            learning_rate=0.00004,
            weight_decay=0.0001,
            batch_size=448,
            train_every=15,
            use_kl_loss=True,
            name="exploration_optimized"
        )
    ]
    
    configs.extend(best_combinations)
    
    return configs

def get_parameter_grid_experiments():
    """Generate a systematic grid search over the most promising parameters."""
    
    # Define parameter grids based on initial findings
    param_grids = {
        'warm_start_min_visits': [60, 75, 90],
        'num_blocks': [22, 25, 28],
        'bottleneck_factor': [3, 4, 5],
        'learning_rate': [0.00002, 0.00003, 0.000035]
    }
    
    configs = []
    
    # Generate all combinations
    for i, (visits, blocks, bottleneck, lr) in enumerate(itertools.product(
        param_grids['warm_start_min_visits'],
        param_grids['num_blocks'], 
        param_grids['bottleneck_factor'],
        param_grids['learning_rate']
    )):
        configs.append(ExperimentConfig(
            warm_start_min_visits=visits,
            num_blocks=blocks,
            bottleneck_factor=bottleneck,
            learning_rate=lr,
            name=f"grid_{i:02d}_v{visits}_b{blocks}_bt{bottleneck}_lr{lr}"
        ))
    
    return configs

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Focused Ultra Deep Parameter Experiments')
    parser.add_argument('--iterations', type=int, default=15000, help='Number of training iterations per experiment')
    parser.add_argument('--mode', type=str, choices=['focused', 'grid', 'combinations'], default='focused',
                       help='Which type of experiments to run')
    args = parser.parse_args()
    
    # Get experiment configurations based on mode
    if args.mode == 'focused':
        configs_to_run = get_focused_experiment_configs()
        print(f"Running {len(configs_to_run)} focused parameter experiments...")
    elif args.mode == 'grid':
        configs_to_run = get_parameter_grid_experiments()
        print(f"Running {len(configs_to_run)} grid search experiments...")
    elif args.mode == 'combinations':
        focused_configs = get_focused_experiment_configs()
        # Only run the best combination experiments
        configs_to_run = [c for c in focused_configs if 'optimized' in c.name]
        print(f"Running {len(configs_to_run)} optimized combination experiments...")
    
    # Update iterations for all configs
    for config in configs_to_run:
        config.num_iterations = args.iterations
    
    print(f"Each experiment will run for {args.iterations} iterations.")
    
    # Run experiments
    runner = ExperimentRunner(base_iterations=args.iterations)
    runner.run_all_experiments(configs_to_run)
    
    print(f"\nAll experiments completed! Results saved in: {runner.experiment_dir}") 
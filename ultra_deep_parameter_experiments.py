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

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be disabled.")
    MATPLOTLIB_AVAILABLE = False

# GPU/Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import the existing components
from kuhn import KuhnGame, KuhnState, Action, Card, card_to_string
from utils import KuhnStrategy, calculate_exploitability
from train2 import (
    get_state_features, UltraDeepNN, count_parameters, 
    convert_mccfr_to_kuhn_strategies, train_neural_networks_on_batch,
    INPUT_SIZE, NUM_TOTAL_ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION
)

# --- KEY PARAMETERS TO EXPERIMENT WITH ---
class ExperimentConfig:
    """Configuration class for ultra_deep experiments."""
    
    def __init__(self, 
                 # MCCFR Parameters
                 warm_start_min_visits=100,
                 training_data_collection_threshold=20,
                 
                 # Network Architecture Parameters  
                 num_blocks=20,
                 bottleneck_factor=4,
                 hidden_size=1536,
                 
                 # Training Parameters
                 learning_rate=0.00003,
                 weight_decay=0.0005,
                 batch_size=384,
                 train_every=25,
                 
                 # Learning Rate Schedule Parameters
                 warmup_ratio=0.1,
                 use_cosine_annealing=True,
                 
                 # Gradient Parameters
                 gradient_clip_norm=1.0,
                 accumulation_steps=1,
                 
                 # Loss Function Parameters
                 use_kl_loss=False,  # vs MSE loss
                 
                 # Training Iterations
                 num_iterations=10000,
                 
                 # Experiment name
                 name="unnamed"):
        
        self.warm_start_min_visits = warm_start_min_visits
        self.training_data_collection_threshold = training_data_collection_threshold
        self.num_blocks = num_blocks
        self.bottleneck_factor = bottleneck_factor
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_every = train_every
        self.warmup_ratio = warmup_ratio
        self.use_cosine_annealing = use_cosine_annealing
        self.gradient_clip_norm = gradient_clip_norm
        self.accumulation_steps = accumulation_steps
        self.use_kl_loss = use_kl_loss
        self.num_iterations = num_iterations
        self.name = name
    
    def to_dict(self):
        return self.__dict__.copy()
    
    def __str__(self):
        return f"Config(visits={self.warm_start_min_visits}, blocks={self.num_blocks}, " \
               f"btl_factor={self.bottleneck_factor}, hidden={self.hidden_size}, " \
               f"lr={self.learning_rate}, wd={self.weight_decay}, bs={self.batch_size})"

# --- PREDEFINED EXPERIMENT CONFIGURATIONS ---
def get_experiment_configs():
    """Define different experiment configurations to test."""
    
    configs = []
    
    # Baseline configuration
    configs.append(ExperimentConfig(name="baseline"))
    
    # Test different WARM_START_MIN_VISITS values
    for visits in [50, 75, 100, 150, 200]:
        configs.append(ExperimentConfig(
            warm_start_min_visits=visits,
            name=f"visits_{visits}"
        ))
    
    # Test different network architectures
    for num_blocks in [10, 15, 20, 25, 30]:
        configs.append(ExperimentConfig(
            num_blocks=num_blocks,
            name=f"blocks_{num_blocks}"
        ))
    
    for bottleneck_factor in [2, 3, 4, 6, 8]:
        configs.append(ExperimentConfig(
            bottleneck_factor=bottleneck_factor,
            name=f"bottleneck_{bottleneck_factor}"
        ))
    
    for hidden_size in [1024, 1280, 1536, 1792, 2048]:
        configs.append(ExperimentConfig(
            hidden_size=hidden_size,
            name=f"hidden_{hidden_size}"
        ))
    
    # Test different training parameters
    for lr in [0.00001, 0.00002, 0.00003, 0.00005, 0.0001]:
        configs.append(ExperimentConfig(
            learning_rate=lr,
            name=f"lr_{lr}"
        ))
    
    for wd in [0.0001, 0.0003, 0.0005, 0.001, 0.003]:
        configs.append(ExperimentConfig(
            weight_decay=wd,
            name=f"wd_{wd}"
        ))
    
    for batch_size in [256, 320, 384, 448, 512]:
        configs.append(ExperimentConfig(
            batch_size=batch_size,
            name=f"bs_{batch_size}"
        ))
    
    for train_every in [15, 20, 25, 30, 35]:
        configs.append(ExperimentConfig(
            train_every=train_every,
            name=f"train_{train_every}"
        ))
    
    # Test combination of promising parameters
    promising_combinations = [
        # Fast convergence focused
        ExperimentConfig(
            warm_start_min_visits=75,
            num_blocks=15,
            learning_rate=0.00005,
            batch_size=512,
            train_every=20,
            name="fast_convergence"
        ),
        
        # High capacity focused
        ExperimentConfig(
            warm_start_min_visits=150,
            num_blocks=30,
            hidden_size=2048,
            bottleneck_factor=2,
            learning_rate=0.00002,
            weight_decay=0.0001,
            batch_size=256,
            train_every=30,
            name="high_capacity"
        ),
        
        # Stable training focused
        ExperimentConfig(
            warm_start_min_visits=200,
            num_blocks=20,
            learning_rate=0.00003,
            weight_decay=0.001,
            batch_size=384,
            train_every=25,
            gradient_clip_norm=0.5,
            warmup_ratio=0.15,
            name="stable_training"
        ),
        
        # Aggressive exploration
        ExperimentConfig(
            warm_start_min_visits=50,
            training_data_collection_threshold=10,
            num_blocks=25,
            learning_rate=0.0001,
            batch_size=448,
            train_every=15,
            use_kl_loss=True,
            name="aggressive_exploration"
        )
    ]
    
    configs.extend(promising_combinations)
    
    # Add name attribute to configs that don't have it
    for i, config in enumerate(configs):
        if not hasattr(config, 'name'):
            config.name = f"config_{i}"
    
    return configs

# --- EXPERIMENT RUNNER ---
class ExperimentRunner:
    """Manages running multiple experiments and collecting results."""
    
    def __init__(self, base_iterations=10000):
        self.base_iterations = base_iterations
        self.results = []
        self.experiment_dir = f"ultra_deep_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Global info_sets for MCCFR
        self.info_sets = {}
    
    def run_single_experiment(self, config: ExperimentConfig, experiment_id: str):
        """Run a single experiment with the given configuration."""
        print(f"\n{'='*80}")
        print(f"Running Experiment: {experiment_id}")
        print(f"Config: {config}")
        print(f"{'='*80}")
        
        # Reset global state
        self.info_sets = {}
        
        # Create networks with the specified configuration
        action_sampler_nn = UltraDeepNN(
            input_size=INPUT_SIZE,
            hidden_size=config.hidden_size,
            num_actions=NUM_TOTAL_ACTIONS,
            num_blocks=config.num_blocks,
            bottleneck_factor=config.bottleneck_factor
        ).to(device)
        
        warm_start_nn = UltraDeepNN(
            input_size=INPUT_SIZE,
            hidden_size=config.hidden_size,
            num_actions=NUM_TOTAL_ACTIONS,
            num_blocks=config.num_blocks,
            bottleneck_factor=config.bottleneck_factor
        ).to(device)
        
        # Count parameters
        action_sampler_params = count_parameters(action_sampler_nn)
        warm_start_params = count_parameters(warm_start_nn)
        total_params = action_sampler_params + warm_start_params
        
        print(f"Total parameters: {total_params:,}")
        
        # Create optimizers
        action_sampler_optimizer = optim.AdamW(
            action_sampler_nn.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay, 
            betas=(0.9, 0.999)
        )
        warm_start_optimizer = optim.AdamW(
            warm_start_nn.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay, 
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate schedulers
        total_training_steps = config.num_iterations // config.train_every
        
        if config.use_cosine_annealing:
            def lr_lambda(current_step):
                if current_step < total_training_steps * config.warmup_ratio:
                    return current_step / (total_training_steps * config.warmup_ratio)
                else:
                    progress = (current_step - total_training_steps * config.warmup_ratio) / (total_training_steps * (1 - config.warmup_ratio))
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            action_sampler_scheduler = optim.lr_scheduler.LambdaLR(action_sampler_optimizer, lr_lambda)
            warm_start_scheduler = optim.lr_scheduler.LambdaLR(warm_start_optimizer, lr_lambda)
        else:
            action_sampler_scheduler = optim.lr_scheduler.CosineAnnealingLR(action_sampler_optimizer, T_max=total_training_steps, eta_min=1e-6)
            warm_start_scheduler = optim.lr_scheduler.CosineAnnealingLR(warm_start_optimizer, T_max=total_training_steps, eta_min=1e-6)
        
        # Run the training
        start_time = time.time()
        result = self.train_mccfr_with_config(
            config, action_sampler_nn, warm_start_nn, 
            action_sampler_optimizer, warm_start_optimizer,
            action_sampler_scheduler, warm_start_scheduler
        )
        training_time = time.time() - start_time
        
        # Compile results
        experiment_result = {
            'experiment_id': experiment_id,
            'config': config.to_dict(),
            'total_parameters': total_params,
            'training_time_seconds': training_time,
            'final_exploitability': result['final_exploitability'],
            'best_exploitability': result['best_exploitability'],
            'exploitability_history': result['exploitability_history'],
            'action_sampler_loss_history': result['action_sampler_loss_history'],
            'warm_start_loss_history': result['warm_start_loss_history'],
            'convergence_iterations': result.get('convergence_iterations', None),
            'total_infosets': len(self.info_sets)
        }
        
        # Save individual experiment result
        result_file = os.path.join(self.experiment_dir, f"{experiment_id}_result.json")
        with open(result_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = experiment_result.copy()
            for key in ['exploitability_history', 'action_sampler_loss_history', 'warm_start_loss_history']:
                if key in serializable_result and serializable_result[key] is not None:
                    serializable_result[key] = [float(x) for x in serializable_result[key]]
            json.dump(serializable_result, f, indent=2)
        
        self.results.append(experiment_result)
        
        print(f"\nExperiment {experiment_id} completed:")
        print(f"  Final Exploitability: {result['final_exploitability']:.6f}")
        print(f"  Best Exploitability: {result['best_exploitability']:.6f}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Total Parameters: {total_params:,}")
        
        return experiment_result
    
    def train_mccfr_with_config(self, config, action_sampler_nn, warm_start_nn, 
                               action_sampler_optimizer, warm_start_optimizer,
                               action_sampler_scheduler, warm_start_scheduler):
        """Train MCCFR with the specified configuration."""
        
        # Store metrics for tracking
        exploitability_history = []
        action_sampler_loss_history = []
        warm_start_loss_history = []
        
        # Training data collection
        training_data = []
        
        game = KuhnGame()
        start_time = time.time()
        
        for t in range(config.num_iterations):
            if t > 0 and t % (config.num_iterations // 10 if config.num_iterations >= 10 else 1) == 0:
                elapsed_time = time.time() - start_time
                progress = t / config.num_iterations
                eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
                print(f"\rIteration {t}/{config.num_iterations} ({progress * 100:.1f}%) - ETA: {eta:.1f}s", end="", flush=True)

            initial_state = game.get_initial_state()
            player_card_map = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}

            # Call the MCCFR sampling function with training data collection
            iteration_training_data = []
            self.mccfr_outcome_sampling_with_config(
                initial_state, player_card_map, inv_reach_prob_sampler=1.0, 
                training_data=iteration_training_data, config=config,
                action_sampler_nn=action_sampler_nn, warm_start_nn=warm_start_nn
            )
            
            # Add collected data to training batch
            training_data.extend(iteration_training_data)

            # Train neural networks periodically
            if len(training_data) >= config.batch_size and t % config.train_every == 0:
                # Sample a batch from collected data
                batch_indices = np.random.choice(len(training_data), config.batch_size, replace=False)
                batch_data = [training_data[i] for i in batch_indices]
                
                # Train networks with the configuration
                action_sampler_loss, warm_start_loss = self.train_neural_networks_with_config(
                    batch_data, config, action_sampler_nn, warm_start_nn,
                    action_sampler_optimizer, warm_start_optimizer
                )
                action_sampler_loss_history.append(action_sampler_loss)
                warm_start_loss_history.append(warm_start_loss)
                
                # Step the learning rate schedulers
                action_sampler_scheduler.step()
                warm_start_scheduler.step()
                
                # GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear old training data to prevent memory buildup
                if len(training_data) > 1000:
                    training_data = training_data[-500:]  # Keep only recent data

            # Calculate and print metrics every 100 iterations
            if t % 100 == 0 and t > 0:  # Start from iteration 100
                if self.info_sets:
                    # Calculate exploitability
                    try:
                        player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(self.info_sets)
                        exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                        exploitability_history.append(exploitability)
                    except Exception as e:
                        print(f"\nIteration {t}: Exploitability: Error ({e})")
                        exploitability_history.append(float('nan'))

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s. Total infosets: {len(self.info_sets)}")
        
        # Calculate final exploitability
        final_exploitability = None
        best_exploitability = None
        convergence_iterations = None
        
        if self.info_sets:
            try:
                player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(self.info_sets)
                final_exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                
                if exploitability_history:
                    best_exploitability = min(x for x in exploitability_history if not np.isnan(x))
                    
                    # Find convergence point (when exploitability drops below 0.01)
                    for i, exp in enumerate(exploitability_history):
                        if not np.isnan(exp) and exp < 0.01:
                            convergence_iterations = (i + 1) * 100  # Multiply by 100 since we measure every 100 iterations
                            break
                
            except Exception as e:
                print(f"Final Exploitability: Error ({e})")
        
        return {
            'final_exploitability': final_exploitability,
            'best_exploitability': best_exploitability,
            'exploitability_history': exploitability_history,
            'action_sampler_loss_history': action_sampler_loss_history,
            'warm_start_loss_history': warm_start_loss_history,
            'convergence_iterations': convergence_iterations
        }
    
    def get_strategy_from_regrets(self, regrets: np.ndarray, legal_actions_indices: list[int]) -> np.ndarray:
        """Same as in train2.py but as instance method."""
        strategy = np.zeros(NUM_TOTAL_ACTIONS)
        if not legal_actions_indices:
            return strategy

        # Filter regrets for legal actions
        legal_regrets = regrets[legal_actions_indices]
        positive_legal_regrets = np.maximum(legal_regrets, 0)
        sum_positive_legal_regrets = np.sum(positive_legal_regrets)

        if sum_positive_legal_regrets > 0:
            normalized_positive_regrets = positive_legal_regrets / sum_positive_legal_regrets
            for i, overall_idx in enumerate(legal_actions_indices):
                strategy[overall_idx] = normalized_positive_regrets[i]
        else:
            # Uniform random for legal actions if all regrets are non-positive
            prob = 1.0 / len(legal_actions_indices)
            for idx in legal_actions_indices:
                strategy[idx] = prob
        return strategy
    
    def get_infoset_key(self, state: KuhnState, player_card: Card) -> tuple[str, tuple[str, ...]]:
        """Same as in train2.py but as instance method."""
        history_action_names = tuple(action.name for action in state._history)
        return (player_card.name, history_action_names)
    
    def mccfr_outcome_sampling_with_config(self, state: KuhnState, player_card_map: dict[int, Card], 
                                         inv_reach_prob_sampler: float, training_data: list, 
                                         config: ExperimentConfig, action_sampler_nn, warm_start_nn):
        """MCCFR outcome sampling with configuration parameters."""
        current_player = state._current_player

        if state.is_terminal():
            return [r * inv_reach_prob_sampler for r in state.get_returns()]

        infoset_key = self.get_infoset_key(state, player_card_map[current_player])
        legal_actions_enums = state.get_legal_actions()
        
        if not legal_actions_enums:
            if state.is_terminal():
                return [r * inv_reach_prob_sampler for r in state.get_returns()]
            else:
                print(f"Warning: No legal actions but state not terminal. State: {state}")
                return [0.0, 0.0]

        legal_actions_indices = sorted([ACTION_TO_IDX[act] for act in legal_actions_enums])

        if infoset_key not in self.info_sets:
            self.info_sets[infoset_key] = {
                'regrets': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
                'strategy_sum': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
                'visits': 0
            }
        
        info_node = self.info_sets[infoset_key]
        info_node['visits'] += 1

        state_features_tensor = get_state_features(state, player_card_map[current_player])

        # Get current strategy using config.warm_start_min_visits
        if info_node['visits'] < config.warm_start_min_visits:
            with torch.no_grad():
                action_probs_all = warm_start_nn(state_features_tensor).squeeze(0).cpu().numpy()
            
            current_strategy = np.zeros(NUM_TOTAL_ACTIONS)
            prob_sum_legal = 0.0
            for idx in legal_actions_indices:
                current_strategy[idx] = action_probs_all[idx]
                prob_sum_legal += action_probs_all[idx]
            
            if prob_sum_legal > 1e-6:
                current_strategy /= prob_sum_legal
            else:
                prob = 1.0 / len(legal_actions_indices)
                for idx in legal_actions_indices:
                    current_strategy[idx] = prob
        else:
            current_strategy = self.get_strategy_from_regrets(info_node['regrets'], legal_actions_indices)

        # Collect training data using config.training_data_collection_threshold
        if training_data is not None and info_node['visits'] >= config.training_data_collection_threshold:
            # Create target strategy for WARM START network
            warm_start_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
            for idx in legal_actions_indices:
                warm_start_target[idx] = current_strategy[idx]
            
            if torch.sum(warm_start_target) > 1e-6:
                warm_start_target = warm_start_target / torch.sum(warm_start_target)
            else:
                for idx in legal_actions_indices:
                    warm_start_target[idx] = 1.0 / len(legal_actions_indices)
            
            # Create target for ACTION SAMPLER network
            action_sampler_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
            current_regrets = info_node['regrets']
            
            regret_scores = np.zeros(NUM_TOTAL_ACTIONS)
            for idx in legal_actions_indices:
                regret_val = current_regrets[idx]
                if regret_val > 0:
                    regret_scores[idx] = regret_val ** 1.5
                else:
                    regret_scores[idx] = 0.1
            
            total_regret_score = sum(regret_scores[idx] for idx in legal_actions_indices)
            if total_regret_score > 1e-9:
                for idx in legal_actions_indices:
                    action_sampler_target[idx] = regret_scores[idx] / total_regret_score
            else:
                for idx in legal_actions_indices:
                    action_sampler_target[idx] = 1.0 / len(legal_actions_indices)
            
            training_data.append((state_features_tensor.squeeze(0).cpu(), warm_start_target.cpu(), action_sampler_target.cpu()))

        # Get exploration policy from ActionSamplerNN
        with torch.no_grad():
            sampler_probs_all = action_sampler_nn(state_features_tensor).squeeze(0).cpu().numpy()

        exploration_probs = np.zeros(NUM_TOTAL_ACTIONS)
        sampler_prob_sum_legal = 0.0
        for idx in legal_actions_indices:
            exploration_probs[idx] = sampler_probs_all[idx]
            sampler_prob_sum_legal += sampler_probs_all[idx]
        
        if sampler_prob_sum_legal > 1e-6:
            for idx in legal_actions_indices:
                exploration_probs[idx] /= sampler_prob_sum_legal
        else:
            prob = 1.0 / len(legal_actions_indices)
            for idx in legal_actions_indices:
                exploration_probs[idx] = prob
                
        # Sample action
        p_values_for_legal_actions = [exploration_probs[i] for i in legal_actions_indices]
        p_sum = sum(p_values_for_legal_actions)
        if p_sum == 0:
            p_values_for_legal_actions = [1.0/len(legal_actions_indices)] * len(legal_actions_indices)
        else:
            p_values_for_legal_actions = [p/p_sum for p in p_values_for_legal_actions]

        sampled_action_idx_in_legal_list = np.random.choice(len(legal_actions_indices), p=p_values_for_legal_actions)
        sampled_action_overall_idx = legal_actions_indices[sampled_action_idx_in_legal_list]
        sampled_action_enum = IDX_TO_ACTION[sampled_action_overall_idx]
        
        prob_of_sampled_action_by_sampler = exploration_probs[sampled_action_overall_idx]
        if prob_of_sampled_action_by_sampler == 0:
            prob_of_sampled_action_by_sampler = 1e-9

        # Recursive call
        new_inv_reach_prob_sampler = inv_reach_prob_sampler / prob_of_sampled_action_by_sampler
        
        weighted_child_utils = self.mccfr_outcome_sampling_with_config(
            state.apply_action(sampled_action_enum), 
            player_card_map,
            new_inv_reach_prob_sampler,
            training_data,
            config,
            action_sampler_nn,
            warm_start_nn
        )
        
        # Regret updates
        payoff_p_from_child_weighted = weighted_child_utils[current_player]
        val_of_sampled_action_path_corrected = payoff_p_from_child_weighted / prob_of_sampled_action_by_sampler
        cfv_I_estimate = current_strategy[sampled_action_overall_idx] * val_of_sampled_action_path_corrected

        for a_idx in legal_actions_indices:
            cfv_I_a_estimate = 0.0
            if a_idx == sampled_action_overall_idx:
                cfv_I_a_estimate = val_of_sampled_action_path_corrected
            
            regret_for_action_a = cfv_I_a_estimate - cfv_I_estimate
            info_node['regrets'][a_idx] += regret_for_action_a

        info_node['strategy_sum'] += current_strategy

        return weighted_child_utils
    
    def train_neural_networks_with_config(self, batch_data, config, action_sampler_nn, warm_start_nn,
                                         action_sampler_optimizer, warm_start_optimizer):
        """Train neural networks with configuration parameters."""
        if not batch_data:
            return 0.0, 0.0
        
        # Unpack batch data
        features_list, warm_start_targets_list, action_sampler_targets_list = zip(*batch_data)
        
        # Convert to tensors
        features_batch = torch.stack(features_list).to(device)
        warm_start_targets_batch = torch.stack(warm_start_targets_list).to(device)
        action_sampler_targets_batch = torch.stack(action_sampler_targets_list).to(device)
        
        # Ensure targets sum to 1.0
        warm_start_targets_batch = warm_start_targets_batch / (torch.sum(warm_start_targets_batch, dim=1, keepdim=True) + 1e-9)
        action_sampler_targets_batch = action_sampler_targets_batch / (torch.sum(action_sampler_targets_batch, dim=1, keepdim=True) + 1e-9)
        
        # Set networks to training mode
        action_sampler_nn.train()
        warm_start_nn.train()
        
        effective_batch_size = len(batch_data) // config.accumulation_steps
        
        total_action_sampler_loss = 0.0
        total_warm_start_loss = 0.0
        
        # Train action sampler network
        action_sampler_optimizer.zero_grad()
        for i in range(config.accumulation_steps):
            start_idx = i * effective_batch_size
            end_idx = (i + 1) * effective_batch_size if i < config.accumulation_steps - 1 else len(batch_data)
            
            mini_features = features_batch[start_idx:end_idx]
            mini_targets = action_sampler_targets_batch[start_idx:end_idx]
            
            action_sampler_pred = action_sampler_nn(mini_features)
            
            # Use configured loss function
            if config.use_kl_loss:
                action_sampler_loss = F.kl_div(
                    torch.log(action_sampler_pred + 1e-9), 
                    mini_targets, 
                    reduction='batchmean'
                )
            else:
                action_sampler_loss = F.mse_loss(action_sampler_pred, mini_targets)
            
            action_sampler_loss = action_sampler_loss / config.accumulation_steps
            action_sampler_loss.backward()
            total_action_sampler_loss += action_sampler_loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(action_sampler_nn.parameters(), max_norm=config.gradient_clip_norm)
        action_sampler_optimizer.step()
        
        # Train warm start network
        warm_start_optimizer.zero_grad()
        for i in range(config.accumulation_steps):
            start_idx = i * effective_batch_size
            end_idx = (i + 1) * effective_batch_size if i < config.accumulation_steps - 1 else len(batch_data)
            
            mini_features = features_batch[start_idx:end_idx]
            mini_targets = warm_start_targets_batch[start_idx:end_idx]
            
            warm_start_pred = warm_start_nn(mini_features)
            
            if config.use_kl_loss:
                warm_start_loss = F.kl_div(
                    torch.log(warm_start_pred + 1e-9), 
                    mini_targets, 
                    reduction='batchmean'
                )
            else:
                warm_start_loss = F.mse_loss(warm_start_pred, mini_targets)
            
            warm_start_loss = warm_start_loss / config.accumulation_steps
            warm_start_loss.backward()
            total_warm_start_loss += warm_start_loss.item()
        
        torch.nn.utils.clip_grad_norm_(warm_start_nn.parameters(), max_norm=config.gradient_clip_norm)
        warm_start_optimizer.step()
        
        # Set networks back to eval mode
        action_sampler_nn.eval()
        warm_start_nn.eval()
        
        return total_action_sampler_loss, total_warm_start_loss
    
    def run_all_experiments(self, configs):
        """Run all experiments in the config list."""
        print(f"Starting {len(configs)} experiments...")
        
        for i, config in enumerate(configs):
            experiment_id = f"{i:03d}_{config.name}" if hasattr(config, 'name') else f"{i:03d}"
            
            try:
                self.run_single_experiment(config, experiment_id)
            except Exception as e:
                print(f"\nError in experiment {experiment_id}: {e}")
                # Continue with next experiment
                continue
        
        # Save summary results
        self.save_summary_results()
        
        # Generate analysis
        self.analyze_results()
    
    def save_summary_results(self):
        """Save summary of all experiment results."""
        summary_file = os.path.join(self.experiment_dir, "experiment_summary.json")
        
        summary_data = {
            'total_experiments': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        for result in self.results:
            # Create a simplified summary
            summary_result = {
                'experiment_id': result['experiment_id'],
                'config_name': result['config'].get('name', 'unknown'),
                'total_parameters': result['total_parameters'],
                'training_time_seconds': result['training_time_seconds'],
                'final_exploitability': result['final_exploitability'],
                'best_exploitability': result['best_exploitability'],
                'convergence_iterations': result['convergence_iterations'],
                'key_config_params': {
                    'warm_start_min_visits': result['config']['warm_start_min_visits'],
                    'num_blocks': result['config']['num_blocks'],
                    'bottleneck_factor': result['config']['bottleneck_factor'],
                    'hidden_size': result['config']['hidden_size'],
                    'learning_rate': result['config']['learning_rate'],
                    'batch_size': result['config']['batch_size']
                }
            }
            summary_data['results'].append(summary_result)
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
    
    def analyze_results(self):
        """Analyze and rank experiment results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        print(f"\n{'='*80}")
        print("EXPERIMENT ANALYSIS")
        print(f"{'='*80}")
        
        # Filter out failed experiments
        valid_results = [r for r in self.results if r['final_exploitability'] is not None]
        
        if not valid_results:
            print("No valid results to analyze.")
            return
        
        # Sort by final exploitability (lower is better)
        valid_results.sort(key=lambda x: x['final_exploitability'])
        
        print(f"\nTOP 10 EXPERIMENTS (by final exploitability):")
        print(f"{'Rank':<4} {'ID':<15} {'Final Exploit.':<12} {'Best Exploit.':<12} {'Params':<10} {'Time(s)':<8}")
        print("-" * 80)
        
        for i, result in enumerate(valid_results[:10]):
            print(f"{i+1:<4} {result['experiment_id']:<15} "
                  f"{result['final_exploitability']:<12.6f} "
                  f"{result['best_exploitability']:<12.6f} "
                  f"{result['total_parameters']:<10,} "
                  f"{result['training_time_seconds']:<8.1f}")
        
        # Analyze parameter correlations
        print(f"\n\nPARAMETER ANALYSIS:")
        self.analyze_parameter_impact(valid_results)
        
        # Save analysis results
        analysis_file = os.path.join(self.experiment_dir, "analysis_results.txt")
        with open(analysis_file, 'w') as f:
            f.write("EXPERIMENT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TOP 10 EXPERIMENTS (by final exploitability):\n")
            for i, result in enumerate(valid_results[:10]):
                f.write(f"{i+1}. {result['experiment_id']}: "
                       f"Final={result['final_exploitability']:.6f}, "
                       f"Best={result['best_exploitability']:.6f}\n")
        
        print(f"\nAnalysis saved to: {analysis_file}")
    
    def analyze_parameter_impact(self, results):
        """Analyze the impact of different parameters on performance."""
        
        # Group results by parameter values
        param_analysis = {}
        
        key_params = ['warm_start_min_visits', 'num_blocks', 'bottleneck_factor', 
                     'hidden_size', 'learning_rate', 'batch_size', 'train_every']
        
        for param in key_params:
            param_groups = {}
            for result in results:
                param_value = result['config'][param]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result['final_exploitability'])
            
            # Calculate average performance for each parameter value
            param_averages = {}
            for value, exploitabilities in param_groups.items():
                param_averages[value] = np.mean(exploitabilities)
            
            # Sort by performance
            sorted_params = sorted(param_averages.items(), key=lambda x: x[1])
            
            print(f"\n{param.upper()} impact (avg final exploitability):")
            for value, avg_exploit in sorted_params[:5]:  # Show top 5
                print(f"  {value}: {avg_exploit:.6f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ultra Deep Parameter Experiments')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations per experiment')
    parser.add_argument('--subset', type=str, choices=['all', 'quick', 'promising'], default='quick',
                       help='Which subset of experiments to run')
    parser.add_argument('--single-param', type=str, help='Run experiments for a single parameter type')
    args = parser.parse_args()
    
    # Get experiment configurations
    all_configs = get_experiment_configs()
    
    if args.subset == 'quick':
        # Run only a subset for quick testing
        configs_to_run = [
            config for config in all_configs 
            if hasattr(config, 'name') and config.name in [
                'baseline', 'visits_75', 'visits_150', 'blocks_15', 'blocks_25',
                'lr_0.00005', 'bs_512', 'fast_convergence', 'stable_training'
            ]
        ]
    elif args.subset == 'promising':
        # Run only the promising combinations
        configs_to_run = [
            config for config in all_configs 
            if hasattr(config, 'name') and config.name in [
                'fast_convergence', 'high_capacity', 'stable_training', 'aggressive_exploration'
            ]
        ]
    elif args.single_param:
        # Run experiments for a single parameter type
        configs_to_run = [
            config for config in all_configs
            if hasattr(config, 'name') and config.name.startswith(args.single_param)
        ]
    else:
        configs_to_run = all_configs
    
    # Update iterations for all configs
    for config in configs_to_run:
        config.num_iterations = args.iterations
    
    print(f"Running {len(configs_to_run)} experiments with {args.iterations} iterations each...")
    
    # Run experiments
    runner = ExperimentRunner(base_iterations=args.iterations)
    runner.run_all_experiments(configs_to_run)
    
    print(f"\nAll experiments completed! Results saved in: {runner.experiment_dir}") 
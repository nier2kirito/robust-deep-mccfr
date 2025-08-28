import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict, deque
import time
import argparse
import json
import os
from datetime import datetime
import itertools
from typing import Dict, List, Tuple, Optional, Any
import warnings

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
from games.kuhn import KuhnGame, KuhnState, Action, Card, card_to_string
from utils import KuhnStrategy, calculate_exploitability
from train import (
    get_state_features, UltraDeepNN, count_parameters, 
    convert_mccfr_to_kuhn_strategies, train_neural_networks_on_batch,
    INPUT_SIZE, NUM_TOTAL_ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION
)

class ExperienceReplayBuffer:
    """Experience replay buffer with prioritized sampling for stable training."""
    
    def __init__(self, max_size: int = 10000, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha  # Prioritization exponent
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.infoset_counts = defaultdict(int)
    
    def add(self, experience: Tuple, priority: float = 1.0, infoset_key: str = None):
        """Add experience with priority."""
        self.buffer.append(experience)
        self.priorities.append(priority)
        if infoset_key:
            self.infoset_counts[infoset_key] += 1
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, List[int]]:
        """Sample batch with prioritized sampling and importance weights."""
        if len(self.buffer) == 0:
            return [], np.array([]), []
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), 
                                 size=min(batch_size, len(self.buffer)), 
                                 p=probs, replace=False)
        
        # Calculate importance weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def get_infoset_priority(self, infoset_key: str) -> float:
        """Get priority based on infoset visit count (less visited = higher priority)."""
        count = self.infoset_counts.get(infoset_key, 0)
        return 1.0 / (count + 1.0)  # Higher priority for less visited infosets

class VarianceEstimator(nn.Module):
    """Neural network to estimate variance of importance-weighted returns."""
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
    
    def forward(self, x):
        return self.network(x)

class DiagnosticTracker:
    """Comprehensive diagnostic tracking for MCCFR training."""
    
    def __init__(self):
        self.metrics = {
            'support_entropy': [],
            'importance_weight_stats': [],
            'strategy_disagreement': [],
            'regret_magnitudes': [],
            'network_utilization': [],
            'exploration_coverage': []
        }
        self.iteration_data = []
    
    def track_support_entropy(self, strategy: np.ndarray, legal_actions: List[int]):
        """Track entropy of action distribution (higher = better exploration)."""
        legal_probs = strategy[legal_actions]
        legal_probs = legal_probs[legal_probs > 1e-9]  # Filter near-zero probabilities
        if len(legal_probs) > 1:
            entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-9))
            self.metrics['support_entropy'].append(entropy)
        else:
            self.metrics['support_entropy'].append(0.0)
    
    def track_importance_weights(self, weights: List[float]):
        """Track statistics of importance weights."""
        if weights:
            stats = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(weights),
                'min': np.min(weights),
                'variance': np.var(weights)
            }
            self.metrics['importance_weight_stats'].append(stats)
    
    def track_strategy_disagreement(self, neural_strategy: np.ndarray, 
                                  regret_strategy: np.ndarray, legal_actions: List[int]):
        """Track disagreement between neural network and regret matching strategies."""
        neural_legal = neural_strategy[legal_actions]
        regret_legal = regret_strategy[legal_actions]
        
        # KL divergence as disagreement measure
        kl_div = np.sum(regret_legal * np.log((regret_legal + 1e-9) / (neural_legal + 1e-9)))
        self.metrics['strategy_disagreement'].append(kl_div)
    
    def get_recent_stats(self, metric: str, window: int = 100) -> Dict:
        """Get recent statistics for a metric."""
        if metric not in self.metrics or not self.metrics[metric]:
            return {}
        
        recent_data = self.metrics[metric][-window:]
        if not recent_data:
            return {}
        
        if metric == 'importance_weight_stats':
            # Special handling for nested statistics
            means = [d['mean'] for d in recent_data]
            variances = [d['variance'] for d in recent_data]
            maxes = [d['max'] for d in recent_data]
            return {
                'avg_mean': np.mean(means),
                'avg_variance': np.mean(variances),
                'max_weight': np.max(maxes)
            }
        else:
            return {
                'mean': np.mean(recent_data),
                'std': np.std(recent_data),
                'recent_trend': np.mean(recent_data[-10:]) - np.mean(recent_data[-20:-10]) if len(recent_data) >= 20 else 0
            }

class RobustMCCFRConfig:
    """Enhanced configuration with risk mitigation parameters."""
    
    def __init__(self,
                 # Original MCCFR Parameters
                 warm_start_min_visits: int = 100,
                 training_data_collection_threshold: int = 20,
                 
                 # Network Architecture
                 num_blocks: int = 20,
                 bottleneck_factor: int = 4,
                 hidden_size: int = 1536,
                 
                 # Training Parameters
                 learning_rate: float = 0.00003,
                 weight_decay: float = 0.0005,
                 batch_size: int = 384,
                 train_every: int = 25,
                 
                 # Risk Mitigation Parameters
                 exploration_epsilon: float = 0.1,  # Uniform exploration mixing
                 importance_weight_clip: float = 10.0,  # Clip importance weights
                 use_target_networks: bool = True,  # Use target networks
                 target_update_freq: int = 100,  # Target network update frequency
                 use_variance_objective: bool = True,  # Train for variance reduction
                 variance_weight: float = 0.1,  # Weight for variance loss
                 baseline_subtraction: bool = True,  # Use baseline subtraction
                 
                 # Experience Replay
                 replay_buffer_size: int = 10000,
                 prioritized_replay: bool = True,
                 replay_alpha: float = 0.6,  # Prioritization strength
                 replay_beta: float = 0.4,  # Importance sampling correction
                 
                 # Regularization
                 entropy_regularization: float = 0.01,
                 label_smoothing: float = 0.1,
                 gradient_clip_norm: float = 1.0,
                 
                 # Training Schedule
                 warmup_ratio: float = 0.1,
                 use_cosine_annealing: bool = True,
                 
                 # Diagnostics
                 diagnostic_freq: int = 100,
                 save_diagnostics: bool = True,
                 
                 # Experiment
                 num_iterations: int = 10000,
                 name: str = "robust_mccfr"):
        
        # Store all parameters
        self.warm_start_min_visits = warm_start_min_visits
        self.training_data_collection_threshold = training_data_collection_threshold
        self.num_blocks = num_blocks
        self.bottleneck_factor = bottleneck_factor
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_every = train_every
        
        # Risk mitigation
        self.exploration_epsilon = exploration_epsilon
        self.importance_weight_clip = importance_weight_clip
        self.use_target_networks = use_target_networks
        self.target_update_freq = target_update_freq
        self.use_variance_objective = use_variance_objective
        self.variance_weight = variance_weight
        self.baseline_subtraction = baseline_subtraction
        
        # Experience replay
        self.replay_buffer_size = replay_buffer_size
        self.prioritized_replay = prioritized_replay
        self.replay_alpha = replay_alpha
        self.replay_beta = replay_beta
        
        # Regularization
        self.entropy_regularization = entropy_regularization
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        
        # Training schedule
        self.warmup_ratio = warmup_ratio
        self.use_cosine_annealing = use_cosine_annealing
        
        # Diagnostics
        self.diagnostic_freq = diagnostic_freq
        self.save_diagnostics = save_diagnostics
        
        # Experiment
        self.num_iterations = num_iterations
        self.name = name
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()

class RobustDeepMCCFR:
    """Robust Deep MCCFR implementation with risk mitigation strategies."""
    
    def __init__(self, config: RobustMCCFRConfig):
        self.config = config
        self.info_sets = {}
        self.iteration = 0
        
        # Initialize networks
        self._init_networks()
        
        # Initialize experience replay
        if config.prioritized_replay:
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=config.replay_buffer_size,
                alpha=config.replay_alpha
            )
        
        # Initialize diagnostics
        self.diagnostics = DiagnosticTracker()
        
        # Baseline estimates per infoset
        self.baselines = {}
        
        # Track training metrics
        self.training_metrics = {
            'policy_losses': [],
            'sampler_losses': [],
            'variance_losses': [],
            'exploitability': []
        }
    
    def _init_networks(self):
        """Initialize all neural networks."""
        # Main networks
        self.policy_net = UltraDeepNN(
            input_size=INPUT_SIZE,
            hidden_size=self.config.hidden_size,
            num_actions=NUM_TOTAL_ACTIONS,
            num_blocks=self.config.num_blocks,
            bottleneck_factor=self.config.bottleneck_factor
        ).to(device)
        
        self.sampler_net = UltraDeepNN(
            input_size=INPUT_SIZE,
            hidden_size=self.config.hidden_size,
            num_actions=NUM_TOTAL_ACTIONS,
            num_blocks=self.config.num_blocks,
            bottleneck_factor=self.config.bottleneck_factor
        ).to(device)
        
        # Target networks (if enabled)
        if self.config.use_target_networks:
            self.target_policy_net = UltraDeepNN(
                input_size=INPUT_SIZE,
                hidden_size=self.config.hidden_size,
                num_actions=NUM_TOTAL_ACTIONS,
                num_blocks=self.config.num_blocks,
                bottleneck_factor=self.config.bottleneck_factor
            ).to(device)
            
            self.target_sampler_net = UltraDeepNN(
                input_size=INPUT_SIZE,
                hidden_size=self.config.hidden_size,
                num_actions=NUM_TOTAL_ACTIONS,
                num_blocks=self.config.num_blocks,
                bottleneck_factor=self.config.bottleneck_factor
            ).to(device)
            
            # Copy weights to target networks
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            self.target_sampler_net.load_state_dict(self.sampler_net.state_dict())
        
        # Variance estimator (if enabled)
        if self.config.use_variance_objective:
            self.variance_estimator = VarianceEstimator(INPUT_SIZE).to(device)
            self.variance_optimizer = optim.AdamW(
                self.variance_estimator.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.sampler_optimizer = optim.AdamW(
            self.sampler_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _update_target_networks(self):
        """Update target networks with current network weights."""
        if not self.config.use_target_networks:
            return
        
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_sampler_net.load_state_dict(self.sampler_net.state_dict())
    
    def _get_exploration_strategy(self, state_features: torch.Tensor, 
                                legal_actions: List[int]) -> np.ndarray:
        """Get exploration strategy with uniform mixing for guaranteed support."""
        with torch.no_grad():
            # Get neural network probabilities
            if self.config.use_target_networks:
                raw_probs = self.target_sampler_net(state_features).squeeze(0).cpu().numpy()
            else:
                raw_probs = self.sampler_net(state_features).squeeze(0).cpu().numpy()
        
        # Extract legal action probabilities
        legal_probs = raw_probs[legal_actions]
        legal_probs = np.maximum(legal_probs, 1e-9)  # Ensure positive
        legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
        
        # Mix with uniform distribution for exploration
        uniform_probs = np.ones(len(legal_actions)) / len(legal_actions)
        mixed_probs = ((1 - self.config.exploration_epsilon) * legal_probs + 
                      self.config.exploration_epsilon * uniform_probs)
        
        # Create full action vector
        exploration_strategy = np.zeros(NUM_TOTAL_ACTIONS)
        for i, action_idx in enumerate(legal_actions):
            exploration_strategy[action_idx] = mixed_probs[i]
        
        return exploration_strategy
    
    def _get_policy_strategy(self, state_features: torch.Tensor, 
                           legal_actions: List[int], 
                           regrets: np.ndarray,
                           visits: int) -> np.ndarray:
        """Get policy strategy with warm-start or regret matching."""
        if visits < self.config.warm_start_min_visits:
            # Use neural network for warm start
            with torch.no_grad():
                if self.config.use_target_networks:
                    raw_probs = self.target_policy_net(state_features).squeeze(0).cpu().numpy()
                else:
                    raw_probs = self.policy_net(state_features).squeeze(0).cpu().numpy()
            
            # Normalize over legal actions
            legal_probs = raw_probs[legal_actions]
            legal_probs = np.maximum(legal_probs, 1e-9)
            legal_probs = legal_probs / np.sum(legal_probs)
            
            strategy = np.zeros(NUM_TOTAL_ACTIONS)
            for i, action_idx in enumerate(legal_actions):
                strategy[action_idx] = legal_probs[i]
        else:
            # Use regret matching
            strategy = self._get_strategy_from_regrets(regrets, legal_actions)
        
        return strategy
    
    def _get_strategy_from_regrets(self, regrets: np.ndarray, legal_actions: List[int]) -> np.ndarray:
        """Convert regrets to strategy using regret matching."""
        strategy = np.zeros(NUM_TOTAL_ACTIONS)
        if not legal_actions:
            return strategy
        
        # Get positive regrets for legal actions
        legal_regrets = regrets[legal_actions]
        positive_regrets = np.maximum(legal_regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            # Proportional to positive regrets
            normalized_regrets = positive_regrets / sum_positive
            for i, action_idx in enumerate(legal_actions):
                strategy[action_idx] = normalized_regrets[i]
        else:
            # Uniform over legal actions
            prob = 1.0 / len(legal_actions)
            for action_idx in legal_actions:
                strategy[action_idx] = prob
        
        return strategy
    
    def _compute_baseline(self, infoset_key: str, current_value: float) -> float:
        """Compute and update baseline for variance reduction."""
        if not self.config.baseline_subtraction:
            return 0.0
        
        if infoset_key not in self.baselines:
            self.baselines[infoset_key] = {'sum': 0.0, 'count': 0}
        
        baseline_info = self.baselines[infoset_key]
        baseline_info['sum'] += current_value
        baseline_info['count'] += 1
        
        return baseline_info['sum'] / baseline_info['count']
    
    def _clip_importance_weight(self, weight: float) -> float:
        """Clip importance weights to prevent extreme values."""
        return np.clip(weight, 1e-6, self.config.importance_weight_clip)
    
    def mccfr_step(self, state: KuhnState, player_cards: Dict[int, Card], 
                   reach_prob: float = 1.0) -> List[float]:
        """Single MCCFR step with robust sampling and regret updates."""
        current_player = state._current_player
        
        if state.is_terminal():
            return [r * reach_prob for r in state.get_returns()]
        
        # Get infoset information
        infoset_key = self._get_infoset_key(state, player_cards[current_player])
        legal_actions = state.get_legal_actions()
        legal_action_indices = [ACTION_TO_IDX[act] for act in legal_actions]
        
        if not legal_action_indices:
            return [0.0, 0.0]
        
        # Initialize infoset if needed
        if infoset_key not in self.info_sets:
            self.info_sets[infoset_key] = {
                'regrets': np.zeros(NUM_TOTAL_ACTIONS),
                'strategy_sum': np.zeros(NUM_TOTAL_ACTIONS),
                'visits': 0
            }
        
        info_node = self.info_sets[infoset_key]
        info_node['visits'] += 1
        
        # Get state features
        state_features = get_state_features(state, player_cards[current_player])
        
        # Get current policy strategy
        policy_strategy = self._get_policy_strategy(
            state_features, legal_action_indices, 
            info_node['regrets'], info_node['visits']
        )
        
        # Get exploration strategy
        exploration_strategy = self._get_exploration_strategy(
            state_features, legal_action_indices
        )
        
        # Track diagnostics
        if self.iteration % self.config.diagnostic_freq == 0:
            self.diagnostics.track_support_entropy(exploration_strategy, legal_action_indices)
            if info_node['visits'] >= self.config.warm_start_min_visits:
                regret_strategy = self._get_strategy_from_regrets(
                    info_node['regrets'], legal_action_indices
                )
                neural_strategy = self._get_policy_strategy(
                    state_features, legal_action_indices, 
                    np.zeros(NUM_TOTAL_ACTIONS), 0  # Force neural network use
                )
                self.diagnostics.track_strategy_disagreement(
                    neural_strategy, regret_strategy, legal_action_indices
                )
        
        # Sample action using exploration strategy
        legal_probs = [exploration_strategy[idx] for idx in legal_action_indices]
        legal_probs = np.array(legal_probs)
        legal_probs = legal_probs / np.sum(legal_probs)  # Ensure normalization
        
        sampled_idx = np.random.choice(len(legal_action_indices), p=legal_probs)
        sampled_action_idx = legal_action_indices[sampled_idx]
        sampled_action = IDX_TO_ACTION[sampled_action_idx]
        
        # Get sampling probability and compute importance weight
        sample_prob = exploration_strategy[sampled_action_idx]
        importance_weight = 1.0 / max(sample_prob, 1e-9)
        clipped_weight = self._clip_importance_weight(importance_weight)
        
        # Track importance weights
        if self.iteration % self.config.diagnostic_freq == 0:
            self.diagnostics.track_importance_weights([importance_weight])
        
        # Recursive call
        child_utilities = self.mccfr_step(
            state.apply_action(sampled_action),
            player_cards,
            reach_prob * policy_strategy[sampled_action_idx]
        )
        
        # Compute utilities with importance weighting
        sampled_utility = child_utilities[current_player] * clipped_weight
        
        # Compute baseline and subtract for variance reduction
        baseline = self._compute_baseline(infoset_key, sampled_utility)
        baseline_corrected_utility = sampled_utility - baseline
        
        # Estimate counterfactual value
        cfv_estimate = policy_strategy[sampled_action_idx] * baseline_corrected_utility
        
        # Update regrets
        for action_idx in legal_action_indices:
            if action_idx == sampled_action_idx:
                action_cfv = baseline_corrected_utility
            else:
                action_cfv = 0.0  # Only observed sampled action
            
            regret = action_cfv - cfv_estimate
            info_node['regrets'][action_idx] += regret
        
        # Update strategy sum
        info_node['strategy_sum'] += policy_strategy
        
        # Collect training data
        if (info_node['visits'] >= self.config.training_data_collection_threshold and
            self.config.prioritized_replay):
            
            # Create training targets
            policy_target = torch.tensor(policy_strategy, dtype=torch.float32)
            sampler_target = torch.tensor(exploration_strategy, dtype=torch.float32)
            
            # Compute priority based on regret magnitude
            regret_magnitude = np.mean(np.abs(info_node['regrets'][legal_action_indices]))
            priority = max(regret_magnitude, 0.1)  # Minimum priority
            
            # Add to replay buffer
            experience = (
                state_features.squeeze(0).cpu(),
                policy_target,
                sampler_target,
                torch.tensor([importance_weight], dtype=torch.float32),
                infoset_key
            )
            self.replay_buffer.add(experience, priority, str(infoset_key))
        
        return child_utilities
    
    def _get_infoset_key(self, state: KuhnState, player_card: Card) -> Tuple:
        """Get unique infoset key."""
        history_actions = tuple(action.name for action in state._history)
        return (player_card.name, history_actions)
    
    def train_networks(self):
        """Train neural networks using experience replay."""
        if not self.config.prioritized_replay or len(self.replay_buffer.buffer) < self.config.batch_size:
            return
        
        # Sample batch from replay buffer
        experiences, importance_weights, indices = self.replay_buffer.sample(
            self.config.batch_size, self.config.replay_beta
        )
        
        if not experiences:
            return
        
        # Unpack experiences
        features_list, policy_targets_list, sampler_targets_list, iw_list, infoset_keys = zip(*experiences)
        
        # Convert to tensors
        features_batch = torch.stack(features_list).to(device)
        policy_targets_batch = torch.stack(policy_targets_list).to(device)
        sampler_targets_batch = torch.stack(sampler_targets_list).to(device)
        importance_weights_tensor = torch.tensor(importance_weights, dtype=torch.float32, device=device)
        
        # Train policy network
        self.policy_net.train()
        self.policy_optimizer.zero_grad()
        
        policy_pred = self.policy_net(features_batch)
        
        # Policy loss with label smoothing and entropy regularization
        if self.config.label_smoothing > 0:
            smooth_targets = ((1 - self.config.label_smoothing) * policy_targets_batch + 
                            self.config.label_smoothing / NUM_TOTAL_ACTIONS)
        else:
            smooth_targets = policy_targets_batch
        
        policy_loss = F.kl_div(
            torch.log(policy_pred + 1e-9), 
            smooth_targets, 
            reduction='none'
        ).sum(dim=1)
        
        # Apply importance weights
        weighted_policy_loss = (policy_loss * importance_weights_tensor).mean()
        
        # Add entropy regularization
        if self.config.entropy_regularization > 0:
            entropy_loss = -(policy_pred * torch.log(policy_pred + 1e-9)).sum(dim=1).mean()
            weighted_policy_loss -= self.config.entropy_regularization * entropy_loss
        
        weighted_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.policy_optimizer.step()
        
        # Train sampler network
        self.sampler_net.train()
        self.sampler_optimizer.zero_grad()
        
        sampler_pred = self.sampler_net(features_batch)
        
        # Sampler loss (standard imitation)
        sampler_loss = F.kl_div(
            torch.log(sampler_pred + 1e-9),
            sampler_targets_batch,
            reduction='none'
        ).sum(dim=1)
        
        weighted_sampler_loss = (sampler_loss * importance_weights_tensor).mean()
        
        # Add variance objective if enabled
        if self.config.use_variance_objective:
            # Estimate variance of importance-weighted returns
            variance_pred_raw = self.variance_estimator(features_batch)
            variance_pred = variance_pred_raw.squeeze(1)
            
            # Target is squared importance weight (proxy for variance)
            iw_stacked = torch.stack(iw_list).to(device)
            iw_squeezed = iw_stacked.squeeze()
            iw_squared = iw_squeezed ** 2
            variance_loss = F.mse_loss(variance_pred, iw_squared.detach())
            
            # Train variance estimator
            self.variance_optimizer.zero_grad()
            variance_loss.backward(retain_graph=True)
            self.variance_optimizer.step()
            
            # Add variance penalty to sampler loss (detach to avoid gradient issues)
            variance_penalty = self.config.variance_weight * variance_pred_raw.detach().mean()
            weighted_sampler_loss += variance_penalty
            
            self.training_metrics['variance_losses'].append(variance_loss.item())
        
        weighted_sampler_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sampler_net.parameters(), self.config.gradient_clip_norm)
        self.sampler_optimizer.step()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            new_policy_errors = F.kl_div(
                torch.log(self.policy_net(features_batch) + 1e-9),
                policy_targets_batch,
                reduction='none'
            ).sum(dim=1).cpu().numpy()
            
            new_priorities = new_policy_errors + 1e-6  # Small epsilon to avoid zero priorities
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Record training metrics
        self.training_metrics['policy_losses'].append(weighted_policy_loss.item())
        self.training_metrics['sampler_losses'].append(weighted_sampler_loss.item())
        
        # Set networks back to eval mode
        self.policy_net.eval()
        self.sampler_net.eval()
    
    def train(self, num_iterations: int) -> Dict[str, Any]:
        """Main training loop."""
        game = KuhnGame()
        start_time = time.time()
        
        # Anomaly detection disabled for performance
        # torch.autograd.set_detect_anomaly(True)
        
        print(f"Starting robust Deep MCCFR training for {num_iterations} iterations...")
        print(f"Configuration: {self.config.name}")
        print(f"Risk mitigation enabled: exploration_eps={self.config.exploration_epsilon}, "
              f"weight_clip={self.config.importance_weight_clip}")
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Get initial game state
            initial_state = game.get_initial_state()
            player_cards = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}
            
            # Run MCCFR step
            self.mccfr_step(initial_state, player_cards)
            
            # Train networks
            if iteration % self.config.train_every == 0 and iteration > 0:
                self.train_networks()
            
            # Update target networks
            if (self.config.use_target_networks and 
                iteration % self.config.target_update_freq == 0 and iteration > 0):
                self._update_target_networks()
            
            # Compute exploitability
            if iteration % 100 == 0 and iteration > 0:
                try:
                    player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(self.info_sets)
                    exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                    self.training_metrics['exploitability'].append(exploitability)
                    
                    if iteration % 1000 == 0:
                        elapsed = time.time() - start_time
                        print(f"Iteration {iteration}: Exploitability = {exploitability:.6f}, "
                              f"Time = {elapsed:.1f}s")
                        
                        # Print diagnostic summary
                        if self.config.save_diagnostics:
                            self._print_diagnostics()
                
                except Exception as e:
                    print(f"Iteration {iteration}: Exploitability computation failed: {e}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        
        # Final exploitability
        final_exploitability = None
        try:
            player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(self.info_sets)
            final_exploitability = calculate_exploitability(player1_strategy, player2_strategy)
        except Exception as e:
            print(f"Final exploitability computation failed: {e}")
        
        return {
            'final_exploitability': final_exploitability,
            'training_time': training_time,
            'training_metrics': self.training_metrics,
            'diagnostics': self.diagnostics.metrics,
            'total_infosets': len(self.info_sets)
        }
    
    def _print_diagnostics(self):
        """Print diagnostic summary."""
        print("\n" + "="*50)
        print("DIAGNOSTIC SUMMARY")
        print("="*50)
        
        # Support entropy
        entropy_stats = self.diagnostics.get_recent_stats('support_entropy')
        if entropy_stats:
            print(f"Support Entropy: mean={entropy_stats['mean']:.3f}, "
                  f"trend={entropy_stats['recent_trend']:.3f}")
        
        # Importance weights
        iw_stats = self.diagnostics.get_recent_stats('importance_weight_stats')
        if iw_stats:
            print(f"Importance Weights: avg_mean={iw_stats['avg_mean']:.3f}, "
                  f"avg_variance={iw_stats['avg_variance']:.3f}, "
                  f"max={iw_stats['max_weight']:.3f}")
        
        # Strategy disagreement
        disagree_stats = self.diagnostics.get_recent_stats('strategy_disagreement')
        if disagree_stats:
            print(f"Strategy Disagreement: mean={disagree_stats['mean']:.3f}, "
                  f"trend={disagree_stats['recent_trend']:.3f}")
        
        print("="*50 + "\n")

def get_ablation_configs() -> List[RobustMCCFRConfig]:
    """Generate systematic ablation study configurations."""
    
    base_config = RobustMCCFRConfig(name="baseline")
    configs = [base_config]
    
    # Individual component ablations
    ablations = [
        # Risk mitigation components
        ("no_exploration_mixing", {"exploration_epsilon": 0.0}),
        ("no_weight_clipping", {"importance_weight_clip": float('inf')}),
        ("no_target_networks", {"use_target_networks": False}),
        ("no_variance_objective", {"use_variance_objective": False}),
        ("no_baseline_subtraction", {"baseline_subtraction": False}),
        ("no_prioritized_replay", {"prioritized_replay": False}),
        
        # Exploration epsilon variations
        ("eps_0.05", {"exploration_epsilon": 0.05}),
        ("eps_0.15", {"exploration_epsilon": 0.15}),
        ("eps_0.20", {"exploration_epsilon": 0.20}),
        
        # Weight clipping variations
        ("clip_5", {"importance_weight_clip": 5.0}),
        ("clip_20", {"importance_weight_clip": 20.0}),
        ("clip_50", {"importance_weight_clip": 50.0}),
        
        # Target network update frequency
        ("target_50", {"target_update_freq": 50}),
        ("target_200", {"target_update_freq": 200}),
        ("target_500", {"target_update_freq": 500}),
        
        # Variance objective weight
        ("var_0.05", {"variance_weight": 0.05}),
        ("var_0.2", {"variance_weight": 0.2}),
        ("var_0.5", {"variance_weight": 0.5}),
    ]
    
    for name, modifications in ablations:
        config = RobustMCCFRConfig(name=name, **modifications)
        configs.append(config)
    
    # Combination studies
    combinations = [
        ("minimal_risk", {
            "exploration_epsilon": 0.05,
            "importance_weight_clip": 5.0,
            "use_target_networks": False,
            "use_variance_objective": False
        }),
        ("maximal_risk", {
            "exploration_epsilon": 0.2,
            "importance_weight_clip": 50.0,
            "use_target_networks": True,
            "target_update_freq": 50,
            "use_variance_objective": True,
            "variance_weight": 0.2
        }),
        ("balanced_approach", {
            "exploration_epsilon": 0.1,
            "importance_weight_clip": 10.0,
            "use_target_networks": True,
            "target_update_freq": 100,
            "use_variance_objective": True,
            "variance_weight": 0.1
        }),
        ("minimal_configuration", {
            "exploration_epsilon": 0.0,
            "importance_weight_clip": float('inf'),
            "use_target_networks": False,
            "use_variance_objective": False,
            "baseline_subtraction": False,
            "prioritized_replay": False
        })
    ]
    
    for name, modifications in combinations:
        config = RobustMCCFRConfig(name=name, **modifications)
        configs.append(config)
    
    return configs

class ExperimentRunner:
    """Run systematic experiments and collect results."""
    
    def __init__(self):
        self.results = []
        self.experiment_dir = f"robust_mccfr_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def run_experiment(self, config: RobustMCCFRConfig, num_iterations: int = 10000) -> Dict:
        """Run a single experiment."""
        print(f"\n{'='*80}")
        print(f"Running experiment: {config.name}")
        print(f"{'='*80}")
        
        # Create and train model
        model = RobustDeepMCCFR(config)
        results = model.train(num_iterations)
        
        # Add configuration info
        results['config'] = config.to_dict()
        results['experiment_name'] = config.name
        
        # Save individual results
        result_file = os.path.join(self.experiment_dir, f"{config.name}_results.json")
        with open(result_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        self.results.append(results)
        
        print(f"Experiment {config.name} completed:")
        print(f"  Final Exploitability: {results.get('final_exploitability', 'N/A')}")
        print(f"  Training Time: {results['training_time']:.1f}s")
        print(f"  Total Infosets: {results['total_infosets']}")
        
        return results
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def run_ablation_study(self, num_iterations: int = 10000):
        """Run complete ablation study."""
        configs = get_ablation_configs()
        
        print(f"Starting ablation study with {len(configs)} configurations...")
        
        for config in configs:
            try:
                self.run_experiment(config, num_iterations)
            except Exception as e:
                print(f"Error in experiment {config.name}: {e}")
                continue
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and summarize experimental results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        print(f"\n{'='*80}")
        print("ABLATION STUDY RESULTS")
        print(f"{'='*80}")
        
        # Filter valid results
        valid_results = [r for r in self.results if r.get('final_exploitability') is not None]
        
        if not valid_results:
            print("No valid results found.")
            return
        
        # Sort by final exploitability
        valid_results.sort(key=lambda x: x['final_exploitability'])
        
        print(f"\nRanking by Final Exploitability:")
        print(f"{'Rank':<4} {'Experiment':<20} {'Final Exploit.':<12} {'Training Time':<12}")
        print("-" * 60)
        
        for i, result in enumerate(valid_results):
            print(f"{i+1:<4} {result['experiment_name']:<20} "
                  f"{result['final_exploitability']:<12.6f} "
                  f"{result['training_time']:<12.1f}")
        
        # Component analysis
        print(f"\n\nComponent Impact Analysis:")
        self._analyze_component_impact(valid_results)
        
        # Save analysis
        analysis_file = os.path.join(self.experiment_dir, "ablation_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write("ABLATION STUDY ANALYSIS\n")
            f.write("="*50 + "\n\n")
            
            f.write("Ranking by Final Exploitability:\n")
            for i, result in enumerate(valid_results):
                f.write(f"{i+1}. {result['experiment_name']}: {result['final_exploitability']:.6f}\n")
        
        print(f"\nAnalysis saved to: {analysis_file}")
    
    def _analyze_component_impact(self, results: List[Dict]):
        """Analyze the impact of individual components."""
        
        # Find baseline result
        baseline_result = next((r for r in results if r['experiment_name'] == 'baseline'), None)
        if not baseline_result:
            print("No baseline result found for comparison.")
            return
        
        baseline_exploitability = baseline_result['final_exploitability']
        
        print(f"Baseline exploitability: {baseline_exploitability:.6f}")
        print(f"\nComponent impact (negative = improvement):")
        
        # Analyze individual ablations
        ablation_results = {}
        for result in results:
            name = result['experiment_name']
            if name.startswith('no_'):
                component = name[3:]  # Remove 'no_' prefix
                impact = result['final_exploitability'] - baseline_exploitability
                ablation_results[component] = impact
                print(f"  {component:<20}: {impact:+.6f}")
        
        # Find best and worst components
        if ablation_results:
            best_component = min(ablation_results.items(), key=lambda x: x[1])
            worst_component = max(ablation_results.items(), key=lambda x: x[1])
            
            print(f"\nMost beneficial component: {best_component[0]} ({best_component[1]:+.6f})")
            print(f"Most harmful when removed: {worst_component[0]} ({worst_component[1]:+.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Robust Deep MCCFR Experiments')
    parser.add_argument('--iterations', type=int, default=10000, 
                       help='Number of training iterations')
    parser.add_argument('--experiment', type=str, choices=['single', 'ablation'], default='ablation',
                       help='Type of experiment to run')
    parser.add_argument('--config', type=str, default='baseline',
                       help='Configuration name for single experiment')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.experiment == 'single':
        # Run single experiment
        config = RobustMCCFRConfig(name=args.config)
        runner.run_experiment(config, args.iterations)
    else:
        # Run ablation study
        runner.run_ablation_study(args.iterations)
    
    print(f"\nAll experiments completed! Results saved in: {runner.experiment_dir}")


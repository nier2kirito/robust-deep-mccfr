"""
Leduc Poker adapter for Robust Deep MCCFR.

This module adapts the existing RobustDeepMCCFR implementation to work with
Leduc Poker instead of Kuhn Poker.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deep_mccfr import RobustMCCFRConfig, RobustDeepMCCFR
from dl_mccfr.games.leduc import LeducGame, LeducState, Card, Action
from dl_mccfr.features_leduc import get_leduc_state_features, LEDUC_INPUT_SIZE, LEDUC_NUM_ACTIONS


class LeducRobustMCCFR(RobustDeepMCCFR):
    """
    Leduc Poker adaptation of Robust Deep MCCFR.
    
    This class extends RobustDeepMCCFR to work with Leduc Poker by:
    - Using Leduc-specific state types and features
    - Adapting action spaces and information sets
    - Using Leduc game mechanics
    """
    
    def __init__(self, config: RobustMCCFRConfig):
        """Initialize Leduc Robust MCCFR."""
        # Don't call super().__init__ because it initializes networks with wrong dimensions
        self.config = config
        self.info_sets = {}
        self.iteration = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Leduc-specific action mappings (all possible actions) - must be defined first
        self.leduc_actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET, Action.RAISE]
        self.action_to_idx = {action: i for i, action in enumerate(self.leduc_actions)}
        self.idx_to_action = {i: action for i, action in enumerate(self.leduc_actions)}
        
        # Initialize Leduc-specific networks
        self._init_leduc_networks()
        
        # Initialize experience replay
        if config.prioritized_replay:
            from deep_mccfr import ExperienceReplayBuffer
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=config.replay_buffer_size,
                alpha=config.replay_alpha
            )
        
        # Initialize diagnostics
        from deep_mccfr import DiagnosticTracker
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
    
    def _init_leduc_networks(self):
        """Initialize networks with Leduc-specific dimensions."""
        from dl_mccfr.networks import create_network
        
        # Use 'ultra_deep' as default network type since config doesn't have network_type
        network_type = 'ultra_deep'
        
        # Use actual number of Leduc actions (5: FOLD, CHECK, CALL, BET, RAISE)
        leduc_num_actions = len(self.leduc_actions)
        
        # Main networks
        self.policy_net = create_network(
            network_type, LEDUC_INPUT_SIZE, leduc_num_actions
        ).to(self.device)
        
        self.sampler_net = create_network(
            network_type, LEDUC_INPUT_SIZE, leduc_num_actions
        ).to(self.device)
        
        # Target networks (if enabled)
        if self.config.use_target_networks:
            self.target_policy_net = create_network(
                network_type, LEDUC_INPUT_SIZE, leduc_num_actions
            ).to(self.device)
            
            self.target_sampler_net = create_network(
                network_type, LEDUC_INPUT_SIZE, leduc_num_actions
            ).to(self.device)
            
            # Copy weights to target networks
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            self.target_sampler_net.load_state_dict(self.sampler_net.state_dict())
        
        # Variance estimator (if enabled)
        if self.config.use_variance_objective:
            from deep_mccfr import VarianceEstimator
            self.variance_estimator = VarianceEstimator(LEDUC_INPUT_SIZE).to(self.device)
            self.variance_optimizer = torch.optim.AdamW(
                self.variance_estimator.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.sampler_optimizer = torch.optim.AdamW(
            self.sampler_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _get_leduc_infoset_key(self, state: LeducState, player_card: Card) -> Tuple:
        """Get unique infoset key for Leduc Poker."""
        # Include both betting history and community card information
        history_actions = tuple(action.name for action in state._history)
        board_card = state._board_cards[0].name if state._board_cards else None
        street = state._street
        return (player_card.name, history_actions, board_card, street)
    
    def _get_exploration_strategy(self, state_features: torch.Tensor, 
                                legal_actions: List[Action]) -> np.ndarray:
        """Get exploration strategy with uniform mixing for guaranteed support."""
        legal_indices = [self.action_to_idx[action] for action in legal_actions]
        
        with torch.no_grad():
            # Get neural network probabilities
            if self.config.use_target_networks:
                raw_probs = self.target_sampler_net(state_features).squeeze(0).cpu().numpy()
            else:
                raw_probs = self.sampler_net(state_features).squeeze(0).cpu().numpy()
        
        # Extract legal action probabilities
        legal_probs = raw_probs[legal_indices]
        legal_probs = np.maximum(legal_probs, 1e-9)  # Ensure positive
        legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
        
        # Mix with uniform distribution for exploration
        uniform_probs = np.ones(len(legal_actions)) / len(legal_actions)
        mixed_probs = ((1 - self.config.exploration_epsilon) * legal_probs + 
                      self.config.exploration_epsilon * uniform_probs)
        
        # Create full action vector
        exploration_strategy = np.zeros(len(self.leduc_actions))
        for i, action in enumerate(legal_actions):
            action_idx = self.action_to_idx[action]
            exploration_strategy[action_idx] = mixed_probs[i]
        
        return exploration_strategy
    
    def _get_policy_strategy(self, state_features: torch.Tensor, 
                           legal_actions: List[Action], 
                           regrets: np.ndarray,
                           visits: int) -> np.ndarray:
        """Get policy strategy with warm-start or regret matching."""
        legal_indices = [self.action_to_idx[action] for action in legal_actions]
        
        if visits < 100:  # Warm start threshold
            # Use neural network for warm start
            with torch.no_grad():
                if self.config.use_target_networks:
                    raw_probs = self.target_policy_net(state_features).squeeze(0).cpu().numpy()
                else:
                    raw_probs = self.policy_net(state_features).squeeze(0).cpu().numpy()
            
            # Normalize over legal actions
            legal_probs = raw_probs[legal_indices]
            legal_probs = np.maximum(legal_probs, 1e-9)
            legal_probs = legal_probs / np.sum(legal_probs)
            
            strategy = np.zeros(len(self.leduc_actions))
            for i, action in enumerate(legal_actions):
                action_idx = self.action_to_idx[action]
                strategy[action_idx] = legal_probs[i]
        else:
            # Use regret matching
            strategy = self._get_strategy_from_regrets(regrets, legal_actions)
        
        return strategy
    
    def _get_strategy_from_regrets(self, regrets: np.ndarray, legal_actions: List[Action]) -> np.ndarray:
        """Convert regrets to strategy using regret matching."""
        strategy = np.zeros(len(self.leduc_actions))
        if not legal_actions:
            return strategy
        
        # Get positive regrets for legal actions
        legal_indices = [self.action_to_idx[action] for action in legal_actions]
        legal_regrets = regrets[legal_indices]
        positive_regrets = np.maximum(legal_regrets, 0)
        sum_positive = np.sum(positive_regrets)
        
        if sum_positive > 0:
            # Proportional to positive regrets
            normalized_regrets = positive_regrets / sum_positive
            for i, action in enumerate(legal_actions):
                action_idx = self.action_to_idx[action]
                strategy[action_idx] = normalized_regrets[i]
        else:
            # Uniform over legal actions
            prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_idx = self.action_to_idx[action]
                strategy[action_idx] = prob
        
        return strategy
    
    def _clip_importance_weight(self, weight: float) -> float:
        """Clip importance weights to prevent extreme values."""
        return np.clip(weight, 1e-6, self.config.importance_weight_clip)
    
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
    
    def mccfr_step(self, state: LeducState, player_cards: Dict[int, Card], 
                   reach_prob: float = 1.0) -> List[float]:
        """Single MCCFR step for Leduc Poker with robust sampling and regret updates."""
        current_player = state._current_player
        
        if state.is_terminal():
            return [r * reach_prob for r in state.get_returns()]
        
        # Get infoset information
        infoset_key = self._get_leduc_infoset_key(state, player_cards[current_player])
        legal_actions = state.get_legal_actions()
        
        if not legal_actions:
            return [0.0, 0.0]
        
        # Initialize infoset if needed
        if infoset_key not in self.info_sets:
            self.info_sets[infoset_key] = {
                'regrets': np.zeros(len(self.leduc_actions)),
                'strategy_sum': np.zeros(len(self.leduc_actions)),
                'visits': 0
            }
        
        info_node = self.info_sets[infoset_key]
        info_node['visits'] += 1
        
        # Get state features
        state_features = get_leduc_state_features(state, current_player)
        state_features = state_features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get current policy strategy
        policy_strategy = self._get_policy_strategy(
            state_features, legal_actions, 
            info_node['regrets'], info_node['visits']
        )
        
        # Get exploration strategy
        exploration_strategy = self._get_exploration_strategy(
            state_features, legal_actions
        )
        
        # Track diagnostics
        if self.iteration % self.config.diagnostic_freq == 0:
            legal_indices = [self.action_to_idx[action] for action in legal_actions]
            self.diagnostics.track_support_entropy(exploration_strategy, legal_indices)
            if info_node['visits'] >= 100:
                regret_strategy = self._get_strategy_from_regrets(
                    info_node['regrets'], legal_actions
                )
                neural_strategy = self._get_policy_strategy(
                    state_features, legal_actions, 
                    np.zeros(len(self.leduc_actions)), 0  # Force neural network use
                )
                self.diagnostics.track_strategy_disagreement(
                    neural_strategy, regret_strategy, legal_indices
                )
        
        # Sample action using exploration strategy
        legal_probs = []
        for action in legal_actions:
            action_idx = self.action_to_idx[action]
            legal_probs.append(exploration_strategy[action_idx])
        
        legal_probs = np.array(legal_probs)
        legal_probs = legal_probs / np.sum(legal_probs)  # Ensure normalization
        
        sampled_idx = np.random.choice(len(legal_actions), p=legal_probs)
        sampled_action = legal_actions[sampled_idx]
        sampled_action_idx = self.action_to_idx[sampled_action]
        
        # Get sampling probability and compute importance weight
        sample_prob = exploration_strategy[sampled_action_idx]
        importance_weight = 1.0 / max(sample_prob, 1e-9)
        clipped_weight = self._clip_importance_weight(importance_weight)
        
        # Track importance weights
        if self.iteration % self.config.diagnostic_freq == 0:
            self.diagnostics.track_importance_weights([importance_weight])
        
        # Apply action and get new state
        new_state = state.apply_action(sampled_action)
        
        # Check if betting round is complete and we need to deal the flop
        if (new_state._street == 0 and len(new_state._history) >= 2):
            last_action = new_state._history[-1]
            prev_action = new_state._history[-2]
            
            # If betting round complete (CHECK-CHECK or BET/RAISE-CALL), deal flop
            if ((last_action == Action.CHECK and prev_action == Action.CHECK) or
                (last_action == Action.CALL and (prev_action == Action.BET or prev_action == Action.RAISE))):
                # Need to get the game instance to deal flop
                game = LeducGame()
                new_state = game.deal_flop(new_state)
        
        # Recursive call
        child_utilities = self.mccfr_step(
            new_state,
            player_cards,
            reach_prob * policy_strategy[sampled_action_idx]
        )
        
        # Compute utilities with importance weighting
        sampled_utility = child_utilities[current_player] * clipped_weight
        
        # Compute baseline and subtract for variance reduction
        baseline = self._compute_baseline(str(infoset_key), sampled_utility)
        baseline_corrected_utility = sampled_utility - baseline
        
        # Estimate counterfactual value
        cfv_estimate = policy_strategy[sampled_action_idx] * baseline_corrected_utility
        
        # Update regrets
        for action in legal_actions:
            action_idx = self.action_to_idx[action]
            if action == sampled_action:
                action_cfv = baseline_corrected_utility
            else:
                action_cfv = 0.0  # Only observed sampled action
            
            regret = action_cfv - cfv_estimate
            info_node['regrets'][action_idx] += regret
        
        # Update strategy sum
        info_node['strategy_sum'] += policy_strategy
        
        # Collect training data
        if (info_node['visits'] >= 20 and self.config.prioritized_replay):
            # Create training targets
            policy_target = torch.tensor(policy_strategy, dtype=torch.float32)
            sampler_target = torch.tensor(exploration_strategy, dtype=torch.float32)
            
            # Compute priority based on regret magnitude
            legal_indices = [self.action_to_idx[action] for action in legal_actions]
            regret_magnitude = np.mean(np.abs(info_node['regrets'][legal_indices]))
            priority = max(regret_magnitude, 0.1)  # Minimum priority
            
            # Add to replay buffer
            experience = (
                state_features.squeeze(0).cpu(),
                policy_target,
                sampler_target,
                torch.tensor([importance_weight], dtype=torch.float32),
                str(infoset_key)
            )
            self.replay_buffer.add(experience, priority, str(infoset_key))
        
        return child_utilities
    
    def get_current_stats(self) -> Dict:
        """Get current diagnostic statistics."""
        if hasattr(self, 'diagnostics'):
            return {
                'support_entropy': self.diagnostics.get_recent_stats('support_entropy'),
                'importance_weights': self.diagnostics.get_recent_stats('importance_weight_stats'),
                'strategy_disagreement': self.diagnostics.get_recent_stats('strategy_disagreement')
            }
        return {}

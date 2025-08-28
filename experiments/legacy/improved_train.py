import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict, deque
import time
import argparse
from typing import Dict, List, Tuple, Optional
import copy

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
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")

# Assuming kuhn.py is in the same directory or accessible via PYTHONPATH
from games.kuhn import KuhnGame, KuhnState, Action, Card, card_to_string

# Import exploitability functions from utils.py
from utils import KuhnStrategy, calculate_exploitability

# --- Enhanced Neural Network Definitions ---
class ImprovedBaseNN(nn.Module):
    """Improved base network with better regularization and architectural improvements."""
    def __init__(self, input_size, hidden_size, num_actions, dropout_rate=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Main network with skip connections
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Skip connection projection if needed
        self.skip_projection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        # Output layers with entropy regularization support
        self.output_fc = nn.Linear(hidden_size, num_actions)
        self.output_dropout = nn.Dropout(dropout_rate * 0.5)  # Less dropout before final layer
        
        # Temperature parameter for entropy control (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, apply_softmax=True, return_logits=False):
        # Input normalization
        x_norm = self.input_norm(x)
        
        # First layer
        h1 = F.gelu(self.norm1(self.fc1(x_norm)))
        h1 = self.dropout1(h1)
        
        # Second layer with skip connection
        h2 = F.gelu(self.norm2(self.fc2(h1)))
        h2 = self.dropout2(h2)
        
        # Skip connection
        skip = self.skip_projection(x_norm)
        h2 = h2 + skip
        
        # Output layer
        logits = self.output_fc(self.output_dropout(h2))
        
        if return_logits:
            return logits
            
        if apply_softmax:
            # Apply temperature scaling for entropy control
            scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
            return F.softmax(scaled_logits, dim=-1)
        else:
            return logits

class ValueNetwork(nn.Module):
    """Value network for baseline subtraction and hybrid estimation."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Output single value
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        x = self.input_norm(x)
        
        h1 = F.gelu(self.norm1(self.fc1(x)))
        h1 = self.dropout1(h1)
        
        h2 = F.gelu(self.norm2(self.fc2(h1)))
        h2 = self.dropout2(h2)
        
        value = self.value_head(h2)
        return value.squeeze(-1)  # Remove last dimension

class ExperienceBuffer:
    """Experience replay buffer with prioritized sampling support."""
    def __init__(self, max_size=10000, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha  # Priority exponent
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_priority = 1.0
        
    def add(self, experience, priority=None):
        """Add experience with optional priority."""
        if priority is None:
            priority = self.max_priority
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size, beta=0.4):
        """Sample batch with importance sampling weights."""
        if len(self.buffer) == 0:
            return [], []
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = priorities ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), 
                                 p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)  # Normalize by max weight
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for specific experiences."""
        for i, priority in zip(indices, priorities):
            if i < len(self.priorities):
                self.priorities[i] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class DiagnosticsTracker:
    """Comprehensive diagnostics and monitoring."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.infoset_stats = defaultdict(dict)
        
    def log_metric(self, name, value, iteration=None):
        """Log a scalar metric."""
        self.metrics[name].append((iteration, value))
    
    def log_infoset_stats(self, infoset_key, stats):
        """Log statistics for a specific infoset."""
        self.infoset_stats[infoset_key].update(stats)
    
    def get_recent_metric(self, name, n=1):
        """Get the most recent n values of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return []
        return self.metrics[name][-n:]
    
    def compute_importance_weight_stats(self, weights):
        """Compute statistics for importance weights."""
        if len(weights) == 0:
            return {}
        
        weights_array = np.array(weights)
        return {
            'mean': np.mean(weights_array),
            'std': np.std(weights_array),
            'max': np.max(weights_array),
            'min': np.min(weights_array),
            'variance': np.var(weights_array)
        }

# Enhanced feature extraction with more strategic context
def get_enhanced_state_features(state: KuhnState, player_card: Card) -> torch.Tensor:
    """Enhanced state feature representation with additional strategic context."""
    features = []
    
    # 1. Player's card (one-hot encoded) - 3 features
    card_features = [0.0] * 3 
    card_features[player_card.value] = 1.0
    features.extend(card_features)
    
    # 2. Position and turn information - 2 features
    is_first_to_act = (len(state._history) % 2 == 0)
    features.append(1.0 if is_first_to_act else 0.0)
    features.append(len(state._history) / 3.0)  # Normalized turn number
    
    # 3. Detailed action history encoding - 10 features
    history = state._history
    
    # History length and patterns
    features.append(len(history) / 3.0)  # Normalized length
    features.append(1.0 if len(history) == 0 else 0.0)  # Initial decision
    features.append(1.0 if history == [Action.CHECK] else 0.0)  # Facing check
    features.append(1.0 if history == [Action.BET] else 0.0)  # Facing bet
    features.append(1.0 if history == [Action.CHECK, Action.CHECK] else 0.0)  # Both checked
    features.append(1.0 if history == [Action.CHECK, Action.BET] else 0.0)  # Check-bet sequence
    features.append(1.0 if history == [Action.BET, Action.CALL] else 0.0)  # Bet-call sequence
    features.append(1.0 if len(history) >= 1 and history[-1] == Action.FOLD else 0.0)  # Previous fold
    
    # Aggression indicators
    num_bets = sum(1 for a in history if a == Action.BET)
    num_checks = sum(1 for a in history if a == Action.CHECK)
    features.append(num_bets / max(1, len(history)))  # Betting frequency
    features.append(num_checks / max(1, len(history)))  # Checking frequency
    
    # 4. Betting situation analysis - 6 features
    current_bet = state._bets[state._current_player]
    opponent_bet = state._bets[1 - state._current_player]
    
    features.append(current_bet / 3.0)  # Normalized current bet
    features.append(opponent_bet / 3.0)  # Normalized opponent bet
    
    bet_to_match = max(0, opponent_bet - current_bet)
    features.append(bet_to_match / 2.0)  # Normalized amount to call
    
    pot_size = sum(state._bets)
    pot_odds = bet_to_match / pot_size if pot_size > 0 else 0.0
    features.append(min(pot_odds, 1.0))  # Pot odds
    
    # Stack depth considerations (remaining betting capacity)
    max_additional_bet = 1  # In Kuhn, max additional bet is 1
    remaining_capacity_self = max(0, max_additional_bet - (current_bet - 1))  # -1 for ante
    remaining_capacity_opp = max(0, max_additional_bet - (opponent_bet - 1))
    features.append(remaining_capacity_self)
    features.append(remaining_capacity_opp)
    
    # 5. Legal action encoding - 4 features
    legal_actions = state.get_legal_actions()
    legal_action_features = [0.0] * 4  # FOLD, CHECK, CALL, BET
    for action in legal_actions:
        legal_action_features[action.value] = 1.0
    features.extend(legal_action_features)
    
    # 6. Strategic context features - 6 features
    can_check = Action.CHECK in legal_actions
    can_bet = Action.BET in legal_actions
    can_call = Action.CALL in legal_actions
    can_fold = Action.FOLD in legal_actions
    
    features.append(1.0 if can_check else 0.0)
    features.append(1.0 if can_bet else 0.0)
    features.append(1.0 if can_call else 0.0)
    features.append(1.0 if can_fold else 0.0)
    
    facing_aggression = opponent_bet > current_bet
    features.append(1.0 if facing_aggression else 0.0)
    
    # Showdown probability estimation
    will_showdown = False
    if len(history) == 1:
        if history[0] == Action.CHECK and Action.CHECK in legal_actions:
            will_showdown = True
        elif history[0] == Action.BET and Action.CALL in legal_actions:
            will_showdown = True
    elif len(history) == 2 and history == [Action.CHECK, Action.BET] and Action.CALL in legal_actions:
        will_showdown = True
    features.append(1.0 if will_showdown else 0.0)
    
    # 7. Card strength relative features - 4 features
    card_strength = player_card.value / 2.0  # Normalize: J=0, Q=0.5, K=1.0
    features.append(card_strength)
    
    # Relative strength given betting action
    if facing_aggression:
        call_comfort = card_strength
        features.append(call_comfort)
    else:
        bet_comfort = card_strength
        features.append(bet_comfort)
    
    # Bluff potential and value betting potential
    bluff_potential = (1.0 - card_strength) if not facing_aggression else 0.0
    value_potential = card_strength if not facing_aggression else card_strength
    features.append(bluff_potential)
    features.append(value_potential)
    
    # Total: 3 + 2 + 10 + 6 + 4 + 6 + 4 = 35 features
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

INPUT_SIZE = 35  # Updated feature size
NUM_TOTAL_ACTIONS = 4

# Enhanced network configurations
NETWORK_CONFIGS = {
    'improved_base': {
        'class': ImprovedBaseNN,
        'hidden_size': 512,
        'kwargs': {'dropout_rate': 0.2}
    }
}

def create_network(network_type: str = 'improved_base'):
    """Create a network based on the specified type."""
    config = NETWORK_CONFIGS[network_type]
    return config['class'](
        input_size=INPUT_SIZE,
        hidden_size=config['hidden_size'],
        num_actions=NUM_TOTAL_ACTIONS,
        **config['kwargs']
    )

class ImprovedMCCFRTrainer:
    """Improved MCCFR trainer with all theoretical enhancements."""
    
    def __init__(self, network_type='improved_base', learning_rate=0.0001, 
                 epsilon_exploration=0.1, target_update_freq=100, 
                 importance_weight_clip=10.0, baseline_learning_rate=0.001):
        
        # Networks
        self.action_sampler = create_network(network_type).to(device)
        self.optimal_strategy = create_network(network_type).to(device)
        self.value_network = ValueNetwork(INPUT_SIZE, 256).to(device)
        
        # Target networks for stability
        self.action_sampler_target = copy.deepcopy(self.action_sampler)
        self.optimal_strategy_target = copy.deepcopy(self.optimal_strategy)
        
        # Freeze target networks initially
        for param in self.action_sampler_target.parameters():
            param.requires_grad = False
        for param in self.optimal_strategy_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.action_sampler_optimizer = optim.AdamW(
            self.action_sampler.parameters(), lr=learning_rate, weight_decay=0.001
        )
        self.optimal_strategy_optimizer = optim.AdamW(
            self.optimal_strategy.parameters(), lr=learning_rate, weight_decay=0.001
        )
        self.value_optimizer = optim.AdamW(
            self.value_network.parameters(), lr=baseline_learning_rate, weight_decay=0.001
        )
        
        # Training parameters
        self.epsilon_exploration = epsilon_exploration
        self.target_update_freq = target_update_freq
        self.importance_weight_clip = importance_weight_clip
        
        # Experience replay
        self.experience_buffer = ExperienceBuffer(max_size=50000)
        
        # Diagnostics
        self.diagnostics = DiagnosticsTracker()
        
        # Training state
        self.training_step = 0
        self.info_sets = {}
        
        # Gradual network rollout parameters
        self.min_visits_for_network = 10
        self.network_blend_schedule = lambda visits: min(1.0, visits / 100.0)
        
    def get_infoset_key(self, state: KuhnState, player_card: Card) -> tuple:
        """Get infoset key."""
        history_action_names = tuple(action.name for action in state._history)
        return (player_card.name, history_action_names)
    
    def get_strategy_from_regrets(self, regrets: np.ndarray, legal_actions_indices: List[int]) -> np.ndarray:
        """Convert regrets to strategy using regret matching."""
        strategy = np.zeros(NUM_TOTAL_ACTIONS)
        if not legal_actions_indices:
            return strategy

        legal_regrets = regrets[legal_actions_indices]
        positive_legal_regrets = np.maximum(legal_regrets, 0)
        sum_positive_legal_regrets = np.sum(positive_legal_regrets)

        if sum_positive_legal_regrets > 0:
            normalized_positive_regrets = positive_legal_regrets / sum_positive_legal_regrets
            for i, overall_idx in enumerate(legal_actions_indices):
                strategy[overall_idx] = normalized_positive_regrets[i]
        else:
            prob = 1.0 / len(legal_actions_indices)
            for idx in legal_actions_indices:
                strategy[idx] = prob
        return strategy
    
    def get_exploration_policy(self, state_features: torch.Tensor, legal_actions_indices: List[int], 
                             use_target=False) -> np.ndarray:
        """Get exploration policy with epsilon-greedy mixing."""
        network = self.action_sampler_target if use_target else self.action_sampler
        
        with torch.no_grad():
            network_probs = network(state_features).squeeze(0).cpu().numpy()
        
        # Extract probabilities for legal actions
        legal_probs = np.zeros(NUM_TOTAL_ACTIONS)
        network_prob_sum = 0.0
        
        for idx in legal_actions_indices:
            legal_probs[idx] = network_probs[idx]
            network_prob_sum += network_probs[idx]
        
        # Normalize network probabilities
        if network_prob_sum > 1e-6:
            for idx in legal_actions_indices:
                legal_probs[idx] /= network_prob_sum
        else:
            # Fallback to uniform
            uniform_prob = 1.0 / len(legal_actions_indices)
            for idx in legal_actions_indices:
                legal_probs[idx] = uniform_prob
        
        # Apply epsilon-greedy mixing for exploration
        uniform_prob = 1.0 / len(legal_actions_indices)
        exploration_policy = np.zeros(NUM_TOTAL_ACTIONS)
        
        for idx in legal_actions_indices:
            exploration_policy[idx] = (
                (1 - self.epsilon_exploration) * legal_probs[idx] + 
                self.epsilon_exploration * uniform_prob
            )
        
        return exploration_policy
    
    def get_current_strategy(self, state_features: torch.Tensor, legal_actions_indices: List[int], 
                           regrets: np.ndarray, visits: int, use_target=False) -> np.ndarray:
        """Get current strategy with gradual network rollout."""
        network = self.optimal_strategy_target if use_target else self.optimal_strategy
        
        # Regret-based strategy
        regret_strategy = self.get_strategy_from_regrets(regrets, legal_actions_indices)
        
        if visits < self.min_visits_for_network:
            return regret_strategy
        
        # Network-based strategy
        with torch.no_grad():
            network_probs = network(state_features).squeeze(0).cpu().numpy()
        
        network_strategy = np.zeros(NUM_TOTAL_ACTIONS)
        network_prob_sum = 0.0
        
        for idx in legal_actions_indices:
            network_strategy[idx] = network_probs[idx]
            network_prob_sum += network_probs[idx]
        
        if network_prob_sum > 1e-6:
            network_strategy /= network_prob_sum
        else:
            network_strategy = regret_strategy.copy()
        
        # Gradual blending
        blend_factor = self.network_blend_schedule(visits)
        current_strategy = (
            (1 - blend_factor) * regret_strategy + 
            blend_factor * network_strategy
        )
        
        return current_strategy
    
    def compute_baseline(self, state_features: torch.Tensor) -> float:
        """Compute value baseline for variance reduction."""
        with torch.no_grad():
            baseline = self.value_network(state_features).item()
        return baseline
    
    def mccfr_outcome_sampling(self, state: KuhnState, player_card_map: Dict[int, Card], 
                             inv_reach_prob_sampler: float, 
                             training_data: Optional[List] = None) -> List[float]:
        """Enhanced MCCFR with all improvements."""
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

        legal_actions_indices = sorted([action.value for action in legal_actions_enums])

        if infoset_key not in self.info_sets:
            self.info_sets[infoset_key] = {
                'regrets': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
                'strategy_sum': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
                'visits': 0,
                'baseline_sum': 0.0,
                'baseline_count': 0
            }
        
        info_node = self.info_sets[infoset_key]
        info_node['visits'] += 1

        state_features = get_enhanced_state_features(state, player_card_map[current_player])

        # Get current strategy with gradual rollout
        current_strategy = self.get_current_strategy(
            state_features, legal_actions_indices, info_node['regrets'], info_node['visits']
        )
        
        # Get exploration policy with epsilon-greedy
        exploration_policy = self.get_exploration_policy(state_features, legal_actions_indices)
        
        # Sample action
        legal_exploration_probs = [exploration_policy[i] for i in legal_actions_indices]
        
        # Ensure probabilities sum to 1
        prob_sum = sum(legal_exploration_probs)
        if prob_sum > 0:
            legal_exploration_probs = [p / prob_sum for p in legal_exploration_probs]
        else:
            uniform_prob = 1.0 / len(legal_actions_indices)
            legal_exploration_probs = [uniform_prob] * len(legal_actions_indices)

        sampled_action_idx = np.random.choice(len(legal_actions_indices), p=legal_exploration_probs)
        sampled_action_overall_idx = legal_actions_indices[sampled_action_idx]
        sampled_action_enum = Action(sampled_action_overall_idx)
        
        prob_of_sampled_action = exploration_policy[sampled_action_overall_idx]
        if prob_of_sampled_action <= 0:
            prob_of_sampled_action = 1e-9

        # Clip importance weight to prevent explosion
        importance_weight = 1.0 / prob_of_sampled_action
        clipped_importance_weight = min(importance_weight, self.importance_weight_clip)
        
        # Recursive call
        new_inv_reach_prob = inv_reach_prob_sampler * clipped_importance_weight
        
        weighted_child_utils = self.mccfr_outcome_sampling(
            state.apply_action(sampled_action_enum), 
            player_card_map,
            new_inv_reach_prob,
            training_data
        )
        
        # Compute baseline for variance reduction
        baseline = self.compute_baseline(state_features)
        
        # Update baseline statistics
        info_node['baseline_sum'] += baseline
        info_node['baseline_count'] += 1
        
        # Regret computation with baseline subtraction
        payoff_p_from_child = weighted_child_utils[current_player]
        val_of_sampled_action = payoff_p_from_child / prob_of_sampled_action
        
        # Apply baseline subtraction
        val_of_sampled_action_with_baseline = val_of_sampled_action - baseline
        
        # Expected value with baseline
        cfv_I_estimate = current_strategy[sampled_action_overall_idx] * val_of_sampled_action_with_baseline

        # Update regrets with baseline-adjusted values
        for a_idx in legal_actions_indices:
            cfv_I_a_estimate = 0.0
            if a_idx == sampled_action_overall_idx:
                cfv_I_a_estimate = val_of_sampled_action_with_baseline
            
            regret_for_action_a = cfv_I_a_estimate - cfv_I_estimate
            info_node['regrets'][a_idx] += regret_for_action_a

        # Update strategy sum
        info_node['strategy_sum'] += current_strategy

        # Collect training data
        if training_data is not None and info_node['visits'] >= 5:
            # Compute variance-aware targets for action sampler
            variance_weight = min(1.0, clipped_importance_weight / self.importance_weight_clip)
            
            # Target for optimal strategy network
            regret_strategy = self.get_strategy_from_regrets(info_node['regrets'], legal_actions_indices)
            optimal_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
            for idx in legal_actions_indices:
                optimal_target[idx] = regret_strategy[idx]
            
            # Target for action sampler (variance-minimizing)
            sampler_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
            
            # Use regret magnitudes to guide exploration
            regret_magnitudes = np.abs(info_node['regrets'])
            exploration_scores = np.zeros(NUM_TOTAL_ACTIONS)
            
            for idx in legal_actions_indices:
                base_score = regret_magnitudes[idx]
                # Add exploration bonus for under-explored actions
                exploration_bonus = 0.1 / max(1, info_node['visits'] ** 0.5)
                exploration_scores[idx] = base_score + exploration_bonus
            
            total_score = sum(exploration_scores[idx] for idx in legal_actions_indices)
            if total_score > 0:
                for idx in legal_actions_indices:
                    sampler_target[idx] = exploration_scores[idx] / total_score
            else:
                uniform_prob = 1.0 / len(legal_actions_indices)
                for idx in legal_actions_indices:
                    sampler_target[idx] = uniform_prob
            
            # Store training data with importance weight for prioritized replay
            priority = abs(regret_for_action_a) + 1e-6  # Use regret magnitude as priority
            
            training_data.append({
                'state_features': state_features.squeeze(0).cpu(),
                'optimal_target': optimal_target.cpu(),
                'sampler_target': sampler_target.cpu(),
                'baseline_target': torch.tensor(val_of_sampled_action, dtype=torch.float32),
                'importance_weight': clipped_importance_weight,
                'variance_weight': variance_weight,
                'priority': priority
            })

        return weighted_child_utils
    
    def update_target_networks(self):
        """Update target networks with current network parameters."""
        self.action_sampler_target.load_state_dict(self.action_sampler.state_dict())
        self.optimal_strategy_target.load_state_dict(self.optimal_strategy.state_dict())
    
    def train_networks_on_batch(self, batch_data: List[Dict], importance_weights: np.ndarray):
        """Train networks on a batch with importance sampling."""
        if not batch_data:
            return 0.0, 0.0, 0.0
        
        # Prepare batch tensors
        features = torch.stack([item['state_features'] for item in batch_data]).to(device)
        optimal_targets = torch.stack([item['optimal_target'] for item in batch_data]).to(device)
        sampler_targets = torch.stack([item['sampler_target'] for item in batch_data]).to(device)
        baseline_targets = torch.stack([item['baseline_target'] for item in batch_data]).to(device)
        
        # Importance weights for prioritized experience replay
        is_weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)
        
        # Train optimal strategy network
        self.optimal_strategy_optimizer.zero_grad()
        optimal_preds = self.optimal_strategy(features)
        
        # KL divergence loss with importance weighting
        optimal_loss = F.kl_div(
            torch.log(optimal_preds + 1e-9), 
            optimal_targets, 
            reduction='none'
        ).sum(dim=1)
        
        # Add entropy regularization
        entropy_bonus = -torch.sum(optimal_preds * torch.log(optimal_preds + 1e-9), dim=1)
        optimal_loss = optimal_loss - 0.01 * entropy_bonus  # Small entropy bonus
        
        weighted_optimal_loss = torch.mean(optimal_loss * is_weights)
        weighted_optimal_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.optimal_strategy.parameters(), max_norm=1.0)
        self.optimal_strategy_optimizer.step()
        
        # Train action sampler network with variance objective
        self.action_sampler_optimizer.zero_grad()
        sampler_preds = self.action_sampler(features)
        
        # Variance-aware loss for action sampler
        sampler_kl_loss = F.kl_div(
            torch.log(sampler_preds + 1e-9), 
            sampler_targets, 
            reduction='none'
        ).sum(dim=1)
        
        # Add variance penalty (encourage more uniform exploration)
        variance_penalty = torch.var(sampler_preds, dim=1)
        sampler_loss = sampler_kl_loss + 0.1 * variance_penalty
        
        # Add entropy bonus for exploration
        sampler_entropy = -torch.sum(sampler_preds * torch.log(sampler_preds + 1e-9), dim=1)
        sampler_loss = sampler_loss - 0.02 * sampler_entropy
        
        weighted_sampler_loss = torch.mean(sampler_loss * is_weights)
        weighted_sampler_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.action_sampler.parameters(), max_norm=1.0)
        self.action_sampler_optimizer.step()
        
        # Train value network for baseline
        self.value_optimizer.zero_grad()
        value_preds = self.value_network(features)
        value_loss = F.mse_loss(value_preds, baseline_targets, reduction='none')
        weighted_value_loss = torch.mean(value_loss * is_weights)
        weighted_value_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        
        return weighted_optimal_loss.item(), weighted_sampler_loss.item(), weighted_value_loss.item()
    
    def train(self, iterations: int, game: KuhnGame, batch_size: int = 256, 
              train_every: int = 50):
        """Main training loop with all enhancements."""
        print(f"Starting enhanced MCCFR training for {iterations} iterations...")
        print(f"Epsilon exploration: {self.epsilon_exploration}")
        print(f"Target update frequency: {self.target_update_freq}")
        print(f"Importance weight clip: {self.importance_weight_clip}")
        
        start_time = time.time()
        
        # Training metrics
        exploitability_history = []
        optimal_loss_history = []
        sampler_loss_history = []
        value_loss_history = []
        
        for t in range(iterations):
            # Progress reporting
            if t > 0 and t % (iterations // 10 if iterations >= 10 else 1) == 0:
                elapsed_time = time.time() - start_time
                progress = t / iterations
                eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
                print(f"\rIteration {t}/{iterations} ({progress * 100:.1f}%) - ETA: {eta:.1f}s", 
                      end="", flush=True)

            # Run MCCFR iteration
            initial_state = game.get_initial_state()
            player_card_map = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}
            
            training_data = []
            self.mccfr_outcome_sampling(
                initial_state, player_card_map, 
                inv_reach_prob_sampler=1.0, 
                training_data=training_data
            )
            
            # Add to experience buffer
            for data in training_data:
                self.experience_buffer.add(data, priority=data['priority'])
            
            # Train networks periodically
            if len(self.experience_buffer) >= batch_size and t % train_every == 0:
                # Sample from experience buffer with prioritized sampling
                batch_data, importance_weights = self.experience_buffer.sample(batch_size, beta=0.4)
                
                # Train networks
                optimal_loss, sampler_loss, value_loss = self.train_networks_on_batch(
                    batch_data, importance_weights
                )
                
                optimal_loss_history.append(optimal_loss)
                sampler_loss_history.append(sampler_loss)
                value_loss_history.append(value_loss)
                
                # Update target networks
                if t % self.target_update_freq == 0:
                    self.update_target_networks()
                    print(f"\nUpdated target networks at iteration {t}")
                
                # Log diagnostics
                self.diagnostics.log_metric('optimal_loss', optimal_loss, t)
                self.diagnostics.log_metric('sampler_loss', sampler_loss, t)
                self.diagnostics.log_metric('value_loss', value_loss, t)
                
                # Compute importance weight statistics
                iw_stats = self.diagnostics.compute_importance_weight_stats(importance_weights)
                for key, value in iw_stats.items():
                    self.diagnostics.log_metric(f'importance_weight_{key}', value, t)
            
            # Evaluate exploitability periodically
            if t % 100 == 0 and t > 0:
                if self.info_sets:
                    try:
                        player1_strategy, player2_strategy = self.convert_to_kuhn_strategies()
                        exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                        exploitability_history.append(exploitability)
                        
                        self.diagnostics.log_metric('exploitability', exploitability, t)
                        
                        # Print recent losses if available
                        recent_optimal = optimal_loss_history[-1] if optimal_loss_history else 0.0
                        recent_sampler = sampler_loss_history[-1] if sampler_loss_history else 0.0
                        recent_value = value_loss_history[-1] if value_loss_history else 0.0
                        
                        print(f"\nIteration {t}: Exploitability: {exploitability:.6f}, "
                              f"Losses - Optimal: {recent_optimal:.4f}, Sampler: {recent_sampler:.4f}, "
                              f"Value: {recent_value:.4f}")
                        
                        # Log importance weight statistics
                        if importance_weights is not None and len(importance_weights) > 0:
                            iw_stats = self.diagnostics.compute_importance_weight_stats(importance_weights)
                            print(f"Importance weights - Mean: {iw_stats.get('mean', 0):.3f}, "
                                  f"Max: {iw_stats.get('max', 0):.3f}, "
                                  f"Std: {iw_stats.get('std', 0):.3f}")
                            
                    except Exception as e:
                        print(f"\nIteration {t}: Exploitability calculation error: {e}")
                        exploitability_history.append(float('nan'))
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s. Total infosets: {len(self.info_sets)}")
        
        # Final exploitability
        if self.info_sets:
            try:
                player1_strategy, player2_strategy = self.convert_to_kuhn_strategies()
                final_exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                print(f"Final Exploitability: {final_exploitability:.6f}")
            except Exception as e:
                print(f"Final Exploitability calculation error: {e}")
        
        return {
            'exploitability_history': exploitability_history,
            'optimal_loss_history': optimal_loss_history,
            'sampler_loss_history': sampler_loss_history,
            'value_loss_history': value_loss_history,
            'final_strategies': self.get_average_strategies(),
            'diagnostics': self.diagnostics
        }
    
    def convert_to_kuhn_strategies(self) -> Tuple[KuhnStrategy, KuhnStrategy]:
        """Convert internal info_sets to KuhnStrategy format."""
        player1_strategy = KuhnStrategy()
        player2_strategy = KuhnStrategy()
        
        for infoset_key, node_data in self.info_sets.items():
            card_name, history_tuple = infoset_key
            card = Card[card_name]
            history_actions = tuple(Action[action_name] for action_name in history_tuple)
            
            is_player_0_turn = len(history_actions) % 2 == 0
            
            strat_sum = node_data['strategy_sum']
            total_sum = np.sum(strat_sum)
            
            if total_sum > 1e-9:
                avg_strategy = strat_sum / total_sum
                
                for action_idx, prob in enumerate(avg_strategy):
                    if prob > 1e-9:
                        action = Action(action_idx)
                        
                        if is_player_0_turn:
                            player1_strategy.set_action_probability(card, history_actions, action, prob)
                        else:
                            player2_strategy.set_action_probability(card, history_actions, action, prob)
        
        return player1_strategy, player2_strategy
    
    def get_average_strategies(self) -> Dict:
        """Get average strategies from accumulated strategy sums."""
        avg_strategies = {}
        for infoset_key, node_data in self.info_sets.items():
            strat_sum = node_data['strategy_sum']
            total_sum = np.sum(strat_sum)
            if total_sum > 1e-9:
                avg_strategies[infoset_key] = strat_sum / total_sum
            else:
                avg_strategies[infoset_key] = np.zeros_like(strat_sum)
        return avg_strategies

def plot_enhanced_training_metrics(results: Dict):
    """Plot comprehensive training metrics."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Plotting disabled.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Exploitability
    if results['exploitability_history']:
        iterations = [100 + i * 100 for i in range(len(results['exploitability_history']))]
        axes[0, 0].plot(iterations, results['exploitability_history'], 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Exploitability')
        axes[0, 0].set_title('Exploitability Over Training')
        axes[0, 0].grid(True, alpha=0.3)
        
        if results['exploitability_history']:
            final_exploit = results['exploitability_history'][-1]
            axes[0, 0].annotate(f'Final: {final_exploit:.6f}', 
                               xy=(iterations[-1], final_exploit),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Network losses
    train_every = 50
    if results['optimal_loss_history']:
        loss_iterations = [train_every + i * train_every for i in range(len(results['optimal_loss_history']))]
        axes[0, 1].plot(loss_iterations, results['optimal_loss_history'], 'g-', linewidth=2, label='Optimal Strategy', alpha=0.8)
        axes[0, 1].plot(loss_iterations[:len(results['sampler_loss_history'])], results['sampler_loss_history'], 'r-', linewidth=2, label='Action Sampler', alpha=0.8)
        axes[0, 1].plot(loss_iterations[:len(results['value_loss_history'])], results['value_loss_history'], 'm-', linewidth=2, label='Value Network', alpha=0.8)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Network Training Losses')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Importance weight statistics
    diagnostics = results['diagnostics']
    if 'importance_weight_mean' in diagnostics.metrics:
        iw_data = diagnostics.metrics['importance_weight_mean']
        if iw_data:
            iw_iterations, iw_means = zip(*iw_data)
            axes[1, 0].plot(iw_iterations, iw_means, 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Mean Importance Weight')
            axes[1, 0].set_title('Importance Weight Statistics')
            axes[1, 0].grid(True, alpha=0.3)
    
    # Training progress summary
    axes[1, 1].text(0.1, 0.9, 'Training Summary:', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold')
    
    summary_text = []
    if results['exploitability_history']:
        initial_exploit = results['exploitability_history'][0]
        final_exploit = results['exploitability_history'][-1]
        improvement = (initial_exploit - final_exploit) / initial_exploit * 100
        summary_text.append(f"Initial Exploitability: {initial_exploit:.6f}")
        summary_text.append(f"Final Exploitability: {final_exploit:.6f}")
        summary_text.append(f"Improvement: {improvement:.2f}%")
    
    if results['optimal_loss_history']:
        summary_text.append(f"Final Optimal Loss: {results['optimal_loss_history'][-1]:.6f}")
        summary_text.append(f"Final Sampler Loss: {results['sampler_loss_history'][-1]:.6f}")
        summary_text.append(f"Final Value Loss: {results['value_loss_history'][-1]:.6f}")
    
    for i, text in enumerate(summary_text):
        axes[1, 1].text(0.1, 0.8 - i * 0.1, text, transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'enhanced_training_metrics.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced MCCFR Training for Kuhn Poker')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for exploration mixing')
    parser.add_argument('--target_update_freq', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--importance_clip', type=float, default=10.0, help='Importance weight clipping threshold')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--train_every', type=int, default=50, help='Training frequency')
    args = parser.parse_args()

    kuhn_game = KuhnGame()
    
    # Initialize enhanced trainer
    trainer = ImprovedMCCFRTrainer(
        network_type='improved_base',
        learning_rate=0.0001,
        epsilon_exploration=args.epsilon,
        target_update_freq=args.target_update_freq,
        importance_weight_clip=args.importance_clip
    )
    
    # Count parameters
    total_params = (
        sum(p.numel() for p in trainer.action_sampler.parameters() if p.requires_grad) +
        sum(p.numel() for p in trainer.optimal_strategy.parameters() if p.requires_grad) +
        sum(p.numel() for p in trainer.value_network.parameters() if p.requires_grad)
    )
    print(f"Total trainable parameters: {total_params:,}")
    
    # Train
    results = trainer.train(
        iterations=args.iterations,
        game=kuhn_game,
        batch_size=args.batch_size,
        train_every=args.train_every
    )
    
    # Plot results
    if MATPLOTLIB_AVAILABLE:
        plot_enhanced_training_metrics(results)
    
    # Print final strategies sample
    print("\n--- Final Average Strategies (Sample) ---")
    final_strategies = results['final_strategies']
    count = 0
    for infoset, strategy_vector in final_strategies.items():
        if count >= 15:
            print("... (and more strategies)")
            break

        player_card_str = infoset[0]
        history_str = "".join(infoset[1]) if infoset[1] else "Root"
        
        strat_display_list = []
        for action_idx, prob in enumerate(strategy_vector):
            if prob > 1e-4:
                action_name = Action(action_idx).name
                strat_display_list.append(f"{action_name}: {prob:.3f}")
        
        if strat_display_list:
            print(f"Infoset: Card {player_card_str}, Hist: '{history_str}' -> "
                  f"Strategy: {{{', '.join(strat_display_list)}}}")
            count += 1

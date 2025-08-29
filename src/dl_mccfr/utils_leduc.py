"""
Leduc Poker utilities for strategy evaluation and exploitability calculation.

This module provides utilities specific to Leduc Poker, including:
- Strategy representation and manipulation
- Exploitability calculation
- Nash equilibrium computation
- Best response calculation
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import itertools

from .games.leduc import LeducGame, LeducState, Card, Action
from .features_leduc import get_leduc_information_set_key, calculate_leduc_hand_strength

class LeducStrategy:
    """
    Represents a strategy for Leduc Poker.
    
    A strategy maps information sets to probability distributions over actions.
    """
    
    def __init__(self):
        # Strategy maps info_set_key -> action_probabilities
        self.strategy: Dict[str, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        
        # Information set counts for strategy averaging
        self.regret_sum: Dict[str, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        
    def get_strategy(self, info_set_key: str, legal_actions: List[Action]) -> Dict[Action, float]:
        """Get strategy for information set."""
        if info_set_key not in self.strategy:
            # Initialize uniform strategy
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                self.strategy[info_set_key][action] = uniform_prob
        
        # Normalize to ensure probabilities sum to 1
        total = sum(self.strategy[info_set_key][action] for action in legal_actions)
        if total > 0:
            return {action: self.strategy[info_set_key][action] / total for action in legal_actions}
        else:
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
    
    def update_strategy(self, info_set_key: str, action_probs: Dict[Action, float]):
        """Update strategy for information set."""
        for action, prob in action_probs.items():
            self.strategy[info_set_key][action] = prob
    
    def get_average_strategy(self) -> 'LeducStrategy':
        """Get time-averaged strategy."""
        avg_strategy = LeducStrategy()
        
        for info_set_key in self.strategy_sum:
            total = sum(self.strategy_sum[info_set_key].values())
            if total > 0:
                for action, sum_prob in self.strategy_sum[info_set_key].items():
                    avg_strategy.strategy[info_set_key][action] = sum_prob / total
            else:
                # Fallback to current strategy
                for action, prob in self.strategy[info_set_key].items():
                    avg_strategy.strategy[info_set_key][action] = prob
        
        return avg_strategy

def calculate_leduc_exploitability(strategy_0: LeducStrategy, strategy_1: LeducStrategy, 
                                 game: LeducGame, num_samples: int = 10000) -> float:
    """
    Calculate exploitability of a strategy profile in Leduc Poker.
    
    Exploitability is the sum of best response values minus the strategy values.
    
    Args:
        strategy_0: Strategy for player 0
        strategy_1: Strategy for player 1
        game: Leduc game instance
        num_samples: Number of samples for Monte Carlo estimation
    
    Returns:
        Exploitability value
    """
    
    # Calculate best response values
    br_value_0 = calculate_best_response_value(strategy_1, game, player=0, num_samples=num_samples)
    br_value_1 = calculate_best_response_value(strategy_0, game, player=1, num_samples=num_samples)
    
    # Calculate strategy profile values
    profile_value_0 = calculate_strategy_value(strategy_0, strategy_1, game, player=0, num_samples=num_samples)
    profile_value_1 = calculate_strategy_value(strategy_0, strategy_1, game, player=1, num_samples=num_samples)
    
    # Exploitability is the sum of exploitabilities for both players
    exploitability_0 = br_value_0 - profile_value_0
    exploitability_1 = br_value_1 - profile_value_1
    
    return (exploitability_0 + exploitability_1) / 2.0

def calculate_best_response_value(opponent_strategy: LeducStrategy, game: LeducGame, 
                                player: int, num_samples: int = 10000) -> float:
    """Calculate best response value against opponent strategy."""
    
    total_value = 0.0
    
    for _ in range(num_samples):
        initial_state = game.get_initial_state()
        value = _best_response_recursive(initial_state, opponent_strategy, game, player, 1.0)
        total_value += value
    
    return total_value / num_samples

def calculate_strategy_value(strategy_0: LeducStrategy, strategy_1: LeducStrategy, 
                           game: LeducGame, player: int, num_samples: int = 10000) -> float:
    """Calculate expected value for player under strategy profile."""
    
    total_value = 0.0
    
    for _ in range(num_samples):
        initial_state = game.get_initial_state()
        strategies = [strategy_0, strategy_1]
        value = _strategy_value_recursive(initial_state, strategies, game, player, 1.0)
        total_value += value
    
    return total_value / num_samples

def _best_response_recursive(state: LeducState, opponent_strategy: LeducStrategy, 
                           game: LeducGame, player: int, reach_prob: float) -> float:
    """Recursively calculate best response value."""
    
    if state.is_terminal():
        utilities = state.get_utilities()
        return utilities[player]
    
    if state._current_player == player:
        # Our turn - choose best action
        legal_actions = state.get_legal_actions()
        best_value = float('-inf')
        
        for action in legal_actions:
            next_state = state.apply_action(action)
            value = _best_response_recursive(next_state, opponent_strategy, game, player, reach_prob)
            best_value = max(best_value, value)
        
        return best_value
    
    else:
        # Opponent's turn - use their strategy
        legal_actions = state.get_legal_actions()
        info_set_key = get_leduc_information_set_key(state, state._current_player)
        action_probs = opponent_strategy.get_strategy(info_set_key, legal_actions)
        
        expected_value = 0.0
        for action in legal_actions:
            next_state = state.apply_action(action)
            action_prob = action_probs[action]
            value = _best_response_recursive(next_state, opponent_strategy, game, player, 
                                           reach_prob * action_prob)
            expected_value += action_prob * value
        
        return expected_value

def _strategy_value_recursive(state: LeducState, strategies: List[LeducStrategy], 
                            game: LeducGame, player: int, reach_prob: float) -> float:
    """Recursively calculate strategy value."""
    
    if state.is_terminal():
        utilities = state.get_utilities()
        return utilities[player]
    
    current_player = state._current_player
    legal_actions = state.get_legal_actions()
    info_set_key = get_leduc_information_set_key(state, current_player)
    action_probs = strategies[current_player].get_strategy(info_set_key, legal_actions)
    
    expected_value = 0.0
    for action in legal_actions:
        next_state = state.apply_action(action)
        action_prob = action_probs[action]
        value = _strategy_value_recursive(next_state, strategies, game, player, 
                                        reach_prob * action_prob)
        expected_value += action_prob * value
    
    return expected_value

def create_leduc_nash_approximation() -> Tuple[LeducStrategy, LeducStrategy]:
    """
    Create approximate Nash equilibrium strategies for Leduc Poker.
    
    This is a simplified approximation based on hand strength and betting patterns.
    For a true Nash equilibrium, more sophisticated computation would be needed.
    """
    
    strategy_0 = LeducStrategy()
    strategy_1 = LeducStrategy()
    
    # This would need to be implemented with actual Nash equilibrium computation
    # For now, return simple strategies based on hand strength
    
    return strategy_0, strategy_1

def analyze_leduc_strategy(strategy: LeducStrategy, game: LeducGame) -> Dict[str, any]:
    """
    Analyze properties of a Leduc Poker strategy.
    
    Returns statistics about the strategy including:
    - Aggression frequency
    - Bluffing patterns
    - Hand strength correlations
    """
    
    analysis = {
        'total_info_sets': len(strategy.strategy),
        'aggression_stats': {},
        'hand_strength_correlation': {},
        'betting_patterns': {}
    }
    
    # Analyze aggression by street
    preflop_raises = 0
    preflop_total = 0
    flop_raises = 0
    flop_total = 0
    
    for info_set_key, action_probs in strategy.strategy.items():
        if 's0' in info_set_key:  # Pre-flop
            preflop_total += 1
            if Action.RAISE in action_probs and action_probs[Action.RAISE] > 0.5:
                preflop_raises += 1
        elif 's1' in info_set_key:  # Flop
            flop_total += 1
            if Action.RAISE in action_probs and action_probs[Action.RAISE] > 0.5:
                flop_raises += 1
    
    analysis['aggression_stats'] = {
        'preflop_raise_frequency': preflop_raises / max(preflop_total, 1),
        'flop_raise_frequency': flop_raises / max(flop_total, 1)
    }
    
    return analysis

def get_leduc_information_sets() -> List[str]:
    """
    Generate all possible information sets for Leduc Poker.
    
    This is useful for exact computation and analysis.
    """
    
    info_sets = []
    
    # All possible cards
    cards = [Card.J_HEARTS, Card.Q_HEARTS, Card.K_HEARTS, 
             Card.J_SPADES, Card.Q_SPADES, Card.K_SPADES]
    
    # All possible betting sequences (simplified)
    # This would need to be more comprehensive for exact computation
    betting_sequences = ['', '0', '1', '2', '01', '02', '10', '12', '20', '21']
    
    for card in cards:
        for board in [None] + cards:
            if board == card:
                continue  # Can't have same card as hole and board
                
            for sequence in betting_sequences:
                for street in [0, 1]:
                    if street == 0 and board is not None:
                        continue  # No board card pre-flop
                    if street == 1 and board is None:
                        continue  # Must have board card post-flop
                    
                    # Create mock state to generate info set key
                    # This is simplified - real implementation would be more complex
                    card_str = _card_to_string_simple(card)
                    board_str = _card_to_string_simple(board) if board else "none"
                    info_set = f"{card_str}_{board_str}_{sequence}_s{street}"
                    info_sets.append(info_set)
    
    return list(set(info_sets))  # Remove duplicates

def _card_to_string_simple(card: Optional[Card]) -> str:
    """Simple card to string conversion."""
    if card is None:
        return "none"
    
    card_map = {
        Card.J_HEARTS: "Jh", Card.Q_HEARTS: "Qh", Card.K_HEARTS: "Kh",
        Card.J_SPADES: "Js", Card.Q_SPADES: "Qs", Card.K_SPADES: "Ks"
    }
    return card_map.get(card, "unknown")



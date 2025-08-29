"""
Feature extraction for Leduc Poker.

This module provides feature extraction for Leduc Poker states,
which is more complex than Kuhn Poker due to:
- Two betting rounds (pre-flop and flop)
- Community card
- More complex betting sequences
- Larger information set space (~936 vs 12 in Kuhn)
"""

import torch
import numpy as np
from typing import Dict, List
from .games.leduc import LeducState, Card

# Leduc Poker constants
LEDUC_INPUT_SIZE = 48  # Larger feature space for Leduc
LEDUC_NUM_ACTIONS = 3  # Fold, Call, Raise
LEDUC_MAX_HISTORY = 8  # Maximum betting sequence length

def get_leduc_state_features(state: LeducState, player: int) -> torch.Tensor:
    """
    Extract features for Leduc Poker state.
    
    Features include:
    - Player's hole card (one-hot encoded)
    - Community card (one-hot encoded, if revealed)
    - Betting history encoding
    - Pot size information
    - Position information
    - Street information (pre-flop vs flop)
    
    Args:
        state: Current Leduc state
        player: Player index (0 or 1)
    
    Returns:
        Feature tensor of size LEDUC_INPUT_SIZE
    """
    features = []
    
    # 1. Player's hole card (6 features: J♥, Q♥, K♥, J♠, Q♠, K♠)
    player_card = state._player_cards[player]
    card_features = [0.0] * 6
    card_idx = _card_to_index(player_card)
    card_features[card_idx] = 1.0
    features.extend(card_features)
    
    # 2. Community card (6 features + 1 for "no community card")
    community_features = [0.0] * 7
    if state._board_cards:
        board_card = state._board_cards[0]
        board_idx = _card_to_index(board_card)
        community_features[board_idx] = 1.0
    else:
        community_features[6] = 1.0  # No community card
    features.extend(community_features)
    
    # 3. Betting history encoding (16 features: 8 positions × 2 actions)
    history_features = [0.0] * 16
    for i, action in enumerate(state._history[-LEDUC_MAX_HISTORY:]):
        if i < LEDUC_MAX_HISTORY:
            if action.value == 0:  # Fold
                history_features[i * 2] = 1.0
            elif action.value == 1:  # Call
                history_features[i * 2 + 1] = 1.0
            # Raise is encoded as [0, 0] (neither fold nor call)
    features.extend(history_features)
    
    # 4. Pot and betting information (8 features)
    total_pot = sum(state._bets)
    player_bet = state._bets[player]
    opponent_bet = state._bets[1 - player]
    
    pot_features = [
        total_pot / 20.0,  # Normalized pot size
        player_bet / 10.0,  # Normalized player bet
        opponent_bet / 10.0,  # Normalized opponent bet
        (opponent_bet - player_bet) / 10.0,  # Bet difference
        min(player_bet, opponent_bet) / 10.0,  # Minimum bet
        max(player_bet, opponent_bet) / 10.0,  # Maximum bet
        float(player_bet == opponent_bet),  # Bets equal
        float(player_bet > opponent_bet)   # Player ahead
    ]
    features.extend(pot_features)
    
    # 5. Position and street information (5 features)
    position_features = [
        float(player == 0),  # Player 0 (first to act pre-flop)
        float(player == 1),  # Player 1
        float(state._current_player == player),  # Current player to act
        float(state._street == 0),  # Pre-flop
        float(state._street == 1)   # Flop
    ]
    features.extend(position_features)
    
    # 6. Game state information (6 features)
    game_features = [
        len(state._history) / 8.0,  # Normalized history length
        float(state.is_terminal()),  # Terminal state
        float(_can_fold(state)),  # Can fold
        float(_can_call(state)),  # Can call
        float(_can_raise(state)),  # Can raise
        _count_raises(state._history) / 4.0  # Normalized raise count
    ]
    features.extend(game_features)
    
    # Ensure we have exactly LEDUC_INPUT_SIZE features
    assert len(features) == LEDUC_INPUT_SIZE, f"Expected {LEDUC_INPUT_SIZE} features, got {len(features)}"
    
    return torch.tensor(features, dtype=torch.float32)

def _card_to_index(card: Card) -> int:
    """Convert card to index for one-hot encoding."""
    card_map = {
        Card.J_HEARTS: 0,
        Card.Q_HEARTS: 1, 
        Card.K_HEARTS: 2,
        Card.J_SPADES: 3,
        Card.Q_SPADES: 4,
        Card.K_SPADES: 5
    }
    return card_map[card]

def _can_fold(state: LeducState) -> bool:
    """Check if current player can fold."""
    if state.is_terminal():
        return False
    
    # Can always fold if there's a bet to call
    player_bet = state._bets[state._current_player]
    opponent_bet = state._bets[1 - state._current_player]
    return opponent_bet > player_bet

def _can_call(state: LeducState) -> bool:
    """Check if current player can call."""
    if state.is_terminal():
        return False
    
    player_bet = state._bets[state._current_player]
    opponent_bet = state._bets[1 - state._current_player]
    return opponent_bet > player_bet

def _can_raise(state: LeducState) -> bool:
    """Check if current player can raise."""
    if state.is_terminal():
        return False
    
    # Check raise limit (typically 2 raises per street)
    raises_this_street = _count_raises(state._history)
    return raises_this_street < 2

def _count_raises(history: List) -> int:
    """Count number of raises in betting history."""
    # This would need to be implemented based on the action encoding
    # For now, return a simple count
    raise_count = 0
    for action in history:
        if hasattr(action, 'value') and action.value == 2:  # Assuming 2 = raise
            raise_count += 1
    return raise_count

def get_leduc_information_set_key(state: LeducState, player: int) -> str:
    """
    Generate information set key for Leduc Poker.
    
    This creates a unique identifier for the information set,
    which includes all information available to the player.
    """
    # Player's hole card
    player_card = state._player_cards[player]
    card_str = _card_to_string(player_card)
    
    # Community card (if any)
    if state._board_cards:
        board_str = _card_to_string(state._board_cards[0])
    else:
        board_str = "none"
    
    # Betting history
    history_str = "".join([str(action.value) for action in state._history])
    
    # Street
    street_str = f"s{state._street}"
    
    return f"{card_str}_{board_str}_{history_str}_{street_str}"

def _card_to_string(card: Card) -> str:
    """Convert card to string representation."""
    card_map = {
        Card.J_HEARTS: "Jh",
        Card.Q_HEARTS: "Qh",
        Card.K_HEARTS: "Kh", 
        Card.J_SPADES: "Js",
        Card.Q_SPADES: "Qs",
        Card.K_SPADES: "Ks"
    }
    return card_map[card]

def calculate_leduc_hand_strength(hole_card: Card, board_card: Card = None) -> float:
    """
    Calculate hand strength for Leduc Poker.
    
    Returns a value between 0 and 1 indicating relative hand strength.
    """
    if board_card is None:
        # Pre-flop: just hole card strength
        card_values = {
            Card.J_HEARTS: 0.2, Card.J_SPADES: 0.2,
            Card.Q_HEARTS: 0.5, Card.Q_SPADES: 0.5,
            Card.K_HEARTS: 0.8, Card.K_SPADES: 0.8
        }
        return card_values[hole_card]
    else:
        # Post-flop: check for pair
        hole_rank = _get_card_rank(hole_card)
        board_rank = _get_card_rank(board_card)
        
        if hole_rank == board_rank:
            # Pair - very strong
            if hole_rank == 'K':
                return 1.0  # Pair of Kings
            elif hole_rank == 'Q':
                return 0.9  # Pair of Queens  
            else:
                return 0.8  # Pair of Jacks
        else:
            # No pair - just high card
            if hole_rank == 'K':
                return 0.6
            elif hole_rank == 'Q':
                return 0.4
            else:
                return 0.2

def _get_card_rank(card: Card) -> str:
    """Get the rank (J, Q, K) of a card."""
    if card in [Card.J_HEARTS, Card.J_SPADES]:
        return 'J'
    elif card in [Card.Q_HEARTS, Card.Q_SPADES]:
        return 'Q'
    else:
        return 'K'



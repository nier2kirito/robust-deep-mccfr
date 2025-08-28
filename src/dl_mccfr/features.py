"""
Feature extraction for game states.

This module provides functions for converting game states into feature vectors
that can be used as input to neural networks.
"""

import torch
from typing import List
from .games.kuhn import KuhnState, Card, Action


def get_state_features(state: KuhnState, player_card: Card, device: torch.device = None) -> torch.Tensor:
    """
    Enhanced state feature representation for Kuhn poker.
    
    Features include:
    - Player's card (one-hot encoded)
    - Position information (who goes first)
    - Detailed action history encoding
    - Legal action availability
    - Betting situation analysis
    - Strategic context features
    
    Args:
        state: Current game state
        player_card: Card held by the current player
        device: PyTorch device for tensor creation
        
    Returns:
        Feature tensor with shape (1, 27)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    features = []
    
    # 1. Player's card (one-hot: J=0, Q=1, K=2) - 3 features
    card_features = [0.0] * 3 
    card_features[player_card.value] = 1.0
    features.extend(card_features)
    
    # 2. Position information - 1 feature
    # In Kuhn poker, position matters significantly
    is_first_to_act = (len(state._history) % 2 == 0)
    features.append(1.0 if is_first_to_act else 0.0)
    
    # 3. Detailed action history encoding - 8 features
    # Encode specific action patterns that matter strategically
    history = state._history
    
    # History length normalized (0, 0.5, 1.0)
    features.append(len(history) / 2.0)
    
    # Specific action pattern indicators
    features.append(1.0 if len(history) == 0 else 0.0)  # Initial decision
    features.append(1.0 if history == [Action.CHECK] else 0.0)  # Facing check
    features.append(1.0 if history == [Action.BET] else 0.0)  # Facing bet
    features.append(1.0 if history == [Action.CHECK, Action.CHECK] else 0.0)  # Both checked
    features.append(1.0 if history == [Action.CHECK, Action.BET] else 0.0)  # Check-bet sequence
    features.append(1.0 if history == [Action.BET, Action.CALL] else 0.0)  # Bet-call sequence
    features.append(1.0 if len(history) >= 1 and history[-1] == Action.FOLD else 0.0)  # Previous fold
    
    # 4. Betting situation analysis - 4 features
    current_bet = state._bets[state._current_player]
    opponent_bet = state._bets[1 - state._current_player]
    
    # Normalized bet amounts (divide by reasonable maximum, which is 2 in Kuhn)
    features.append(current_bet / 2.0)
    features.append(opponent_bet / 2.0)
    
    # Bet difference and pot odds
    bet_to_match = max(0, opponent_bet - current_bet)
    features.append(bet_to_match / 2.0)  # Normalized amount to call
    
    # Pot odds calculation: how much to win vs how much to pay
    pot_size = sum(state._bets)
    pot_odds = bet_to_match / pot_size if pot_size > 0 else 0.0
    features.append(min(pot_odds, 1.0))  # Cap at 1.0 for normalization
    
    # 5. Legal action encoding - 4 features
    # One-hot encode which actions are legal
    legal_actions = state.get_legal_actions()
    legal_action_features = [0.0] * 4  # FOLD, CHECK, CALL, BET
    for action in legal_actions:
        legal_action_features[action.value] = 1.0
    features.extend(legal_action_features)
    
    # 6. Strategic context features - 4 features
    # These capture important strategic considerations in Kuhn poker
    
    # Can we check (no bet to call)?
    can_check = Action.CHECK in legal_actions
    features.append(1.0 if can_check else 0.0)
    
    # Are we facing aggression (opponent has bet more)?
    facing_aggression = opponent_bet > current_bet
    features.append(1.0 if facing_aggression else 0.0)
    
    # Initiative indicator (did we make the last aggressive action?)
    has_initiative = False
    if len(history) > 0:
        for i in range(len(history) - 1, -1, -1):
            if history[i] == Action.BET:
                # Check if it was our action (considering alternating turns)
                action_player = i % 2
                current_player = state._current_player
                # The player who would act now is the opposite of who acted last
                last_actor = 1 - current_player
                has_initiative = (action_player == last_actor)
                break
    features.append(1.0 if has_initiative else 0.0)
    
    # Showdown indicator (will this lead to showdown if we don't fold?)
    # In Kuhn, after CHECK-CHECK or BET-CALL, it's showdown
    will_showdown = False
    if len(history) == 1:
        if history[0] == Action.CHECK and Action.CHECK in legal_actions:
            will_showdown = True  # CHECK-CHECK leads to showdown
        elif history[0] == Action.BET and Action.CALL in legal_actions:
            will_showdown = True  # BET-CALL leads to showdown
    elif len(history) == 2 and history == [Action.CHECK, Action.BET] and Action.CALL in legal_actions:
        will_showdown = True  # CHECK-BET-CALL leads to showdown
    features.append(1.0 if will_showdown else 0.0)
    
    # 7. Card strength relative to betting action - 3 features
    # Encode how strong our card is in different contexts
    card_strength = player_card.value / 2.0  # Normalize: J=0, Q=0.5, K=1.0
    
    # Raw card strength
    features.append(card_strength)
    
    # Card strength relative to aggressive play
    # Strong cards (K) should bet/call more, weak cards (J) should fold more
    if facing_aggression:
        # How comfortable should we be calling with this card?
        call_comfort = card_strength  # K=1.0 very comfortable, J=0.0 very uncomfortable
        features.append(call_comfort)
    else:
        # How much should we bet with this card?
        bet_comfort = card_strength  # K=1.0 always bet, J=0.0 rarely bet
        features.append(bet_comfort)
    
    # Bluff potential (inverse of card strength for betting)
    # Sometimes we want to bluff with weak cards
    bluff_potential = 1.0 - card_strength if not facing_aggression else 0.0
    features.append(bluff_potential)
    
    # Total: 3 + 1 + 8 + 4 + 4 + 4 + 3 = 27 features
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension


# Constants for feature extraction
INPUT_SIZE = 27
NUM_TOTAL_ACTIONS = 4  # Corresponds to Action enum: FOLD, CHECK, CALL, BET

# Action mappings
ACTION_TO_IDX = {action: action.value for action in Action}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

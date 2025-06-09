import torch
import torch.nn as nn
from kuhn import KuhnGame, Card, Action

def old_meaningless_features(state, player_card):
    """Old approach: meaningless categorical features."""
    features = []
    
    # One-hot encoding for cards (meaningless)
    card_features = [0.0] * 3 
    card_features[player_card.value] = 1.0
    features.extend(card_features)
    
    # Other meaningless categorical indicators
    features.append(float(len(state._history)))
    features.append(float(sum(state._bets)))
    
    return torch.tensor(features, dtype=torch.float32)

def new_meaningful_features(state, player_card):
    """New approach: meaningful numerical features."""
    features = []
    
    # Card as numerical strength (meaningful!)
    card_strength = (player_card.value + 1) / 3.0  # J=0.33, Q=0.67, K=1.0
    features.append(card_strength)
    
    # Action powers (meaningful aggression levels)
    action_powers = {'FOLD': 0.0, 'CHECK': 0.25, 'CALL': 0.5, 'BET': 1.0}
    if state._history:
        last_action_power = action_powers.get(state._history[-1].name, 0.0)
    else:
        last_action_power = 0.0
    features.append(last_action_power)
    
    # Meaningful betting ratios
    total_pot = sum(state._bets)
    investment_ratio = state._bets[state._current_player] / max(total_pot, 1)
    features.append(investment_ratio)
    
    return torch.tensor(features, dtype=torch.float32)

def demonstrate_feature_comparison():
    """Show the dramatic difference between approaches."""
    game = KuhnGame()
    state = game.get_initial_state()
    
    print("=== FEATURE REPRESENTATION COMPARISON ===\n")
    
    for card in [Card.JACK, Card.QUEEN, Card.KING]:
        print(f"--- {card.name} ---")
        
        old_features = old_meaningless_features(state, card)
        new_features = new_meaningful_features(state, card)
        
        print(f"OLD (meaningless): {old_features.tolist()}")
        print(f"  J=[1,0,0], Q=[0,1,0], K=[0,0,1] - No relationship!")
        
        print(f"NEW (meaningful): {new_features.tolist()}")
        print(f"  J=0.33, Q=0.67, K=1.0 - Clear strength ordering!")
        print()
    
    print("WHY MEANINGFUL FEATURES WORK BETTER:")
    print("✓ Neural networks can learn gradients: stronger cards → higher values")
    print("✓ Relationships are built into the representation")
    print("✓ No need to learn that K > Q > J from scratch")
    print("✓ Faster convergence and better generalization")
    print("✓ More sample-efficient learning")
    
    print("\nACTION POWER DEMONSTRATION:")
    actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET]
    action_powers = [0.0, 0.25, 0.5, 1.0]
    
    for action, power in zip(actions, action_powers):
        print(f"{action.name}: {power} (aggression level)")
    
    print("\nKey insight: The neural network can now understand that:")
    print("- BET is more aggressive than CHECK")
    print("- KING is stronger than JACK") 
    print("- These relationships are encoded in the numbers!")

if __name__ == "__main__":
    demonstrate_feature_comparison() 
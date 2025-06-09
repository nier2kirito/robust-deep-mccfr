import random
import numpy as np
from collections import defaultdict
import time
import argparse  # Add argparse for command line arguments

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be disabled.")
    MATPLOTLIB_AVAILABLE = False

# Assuming kuhn.py is in the same directory or accessible via PYTHONPATH
from kuhn import KuhnGame, KuhnState, Action, Card, card_to_string

# Import exploitability functions from utils.py
from utils import KuhnStrategy, calculate_exploitability

# --- Classical MCCFR Logic ---
info_sets = {} # Global store for infoset data: infoset_key -> {regrets, strategy_sum, visits}

# Map Action enum to a fixed index range [0, NUM_TOTAL_ACTIONS-1]
NUM_TOTAL_ACTIONS = 4 # Corresponds to Action enum: FOLD, CHECK, CALL, BET
ACTION_TO_IDX = {action: action.value for action in Action}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

def get_infoset_key(state: KuhnState, player_card: Card) -> tuple[str, tuple[str, ...]]:
    # Infoset: player's card + history of public actions from their perspective
    history_action_names = tuple(action.name for action in state._history)
    return (player_card.name, history_action_names)

def get_strategy_from_regrets(regrets: np.ndarray, legal_actions_indices: list[int]) -> np.ndarray:
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

def classical_mccfr_outcome_sampling(state: KuhnState, player_card_map: dict[int, Card], inv_reach_prob_sampler: float):
    current_player = state._current_player

    if state.is_terminal():
        # Returns utilities weighted by the inverse of the sampling probability of the path taken so far
        return [r * inv_reach_prob_sampler for r in state.get_returns()]

    infoset_key = get_infoset_key(state, player_card_map[current_player])
    legal_actions_enums = state.get_legal_actions()
    
    # Ensure there are legal actions; otherwise, it's effectively terminal for this player.
    if not legal_actions_enums:
         # This case should ideally be caught by state.is_terminal() if game logic is complete
         if state.is_terminal():
             return [r * inv_reach_prob_sampler for r in state.get_returns()]
         else:
             # Fallback: if no legal actions but not terminal, return neutral utilities
             print(f"Warning: No legal actions but state not terminal. State: {state}")
             return [0.0, 0.0]  # Return neutral utilities for both players

    legal_actions_indices = sorted([ACTION_TO_IDX[act] for act in legal_actions_enums])

    if infoset_key not in info_sets:
        info_sets[infoset_key] = {
            'regrets': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
            'strategy_sum': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
            'visits': 0
        }
    
    info_node = info_sets[infoset_key]
    info_node['visits'] += 1

    # Get current strategy (sigma_t) for regret updates using classical regret matching
    current_strategy = get_strategy_from_regrets(info_node['regrets'], legal_actions_indices)

    # Use uniform sampling for exploration (classical MCCFR approach)
    exploration_probs = np.zeros(NUM_TOTAL_ACTIONS)
    prob = 1.0 / len(legal_actions_indices)
    for idx in legal_actions_indices:
        exploration_probs[idx] = prob
            
    # Sample action using uniform exploration
    p_values_for_legal_actions = [exploration_probs[i] for i in legal_actions_indices]
    
    sampled_action_idx_in_legal_list = np.random.choice(len(legal_actions_indices), p=p_values_for_legal_actions)
    sampled_action_overall_idx = legal_actions_indices[sampled_action_idx_in_legal_list]
    sampled_action_enum = IDX_TO_ACTION[sampled_action_overall_idx]
    
    prob_of_sampled_action_by_sampler = exploration_probs[sampled_action_overall_idx]
    if prob_of_sampled_action_by_sampler == 0: # Avoid division by zero
        prob_of_sampled_action_by_sampler = 1e-9 # Small epsilon

    # Recursively call MCCFR, update inverse reach probability
    new_inv_reach_prob_sampler = inv_reach_prob_sampler / prob_of_sampled_action_by_sampler
    
    weighted_child_utils = classical_mccfr_outcome_sampling(
        state.apply_action(sampled_action_enum), 
        player_card_map,
        new_inv_reach_prob_sampler
    )
    
    # Regret and strategy sum updates for the current player
    payoff_p_from_child_weighted = weighted_child_utils[current_player]
    
    # Corrected counterfactual value for the sampled action path from current infoset
    val_of_sampled_action_path_corrected = payoff_p_from_child_weighted / prob_of_sampled_action_by_sampler

    # Expected value of the infoset under current_strategy, using the single sample
    cfv_I_estimate = current_strategy[sampled_action_overall_idx] * val_of_sampled_action_path_corrected

    for a_idx in legal_actions_indices:
        cfv_I_a_estimate = 0.0
        if a_idx == sampled_action_overall_idx:
            cfv_I_a_estimate = val_of_sampled_action_path_corrected
        
        regret_for_action_a = cfv_I_a_estimate - cfv_I_estimate
        info_node['regrets'][a_idx] += regret_for_action_a

    info_node['strategy_sum'] += current_strategy

    return weighted_child_utils


def train_classical_mccfr(iterations: int, game: KuhnGame):
    global info_sets
    info_sets = {} # Reset for each training run
    start_time = time.time()  # Track start time

    # Store metrics for tracking
    exploitability_history = []
    
    for t in range(iterations):
        if t > 0 and t % (iterations // 10 if iterations >= 10 else 1) == 0:
            elapsed_time = time.time() - start_time
            progress = t / iterations
            eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
            print(f"\rIteration {t}/{iterations} ({progress * 100:.1f}%) - ETA: {eta:.1f}s", end="", flush=True)

        initial_state = game.get_initial_state()
        player_card_map = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}

        # Call the classical MCCFR sampling function
        classical_mccfr_outcome_sampling(initial_state, player_card_map, inv_reach_prob_sampler=1.0)

        # Calculate and print metrics every 100 iterations
        if t % 100 == 0 and t > 0:  # Start from iteration 100
            if info_sets:
                # Calculate exploitability
                try:
                    player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(info_sets)
                    exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                    exploitability_history.append(exploitability)
                    
                    print(f"\nIteration {t}: Exploitability: {exploitability:.6f}")
                except Exception as e:
                    print(f"\nIteration {t}: Exploitability: Error ({e})")
                    exploitability_history.append(float('nan'))

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s. Total infosets: {len(info_sets)}")
    
    # Calculate final exploitability
    final_exploitability = None
    if info_sets:
        try:
            player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(info_sets)
            final_exploitability = calculate_exploitability(player1_strategy, player2_strategy)
            print(f"Final Exploitability: {final_exploitability:.6f}")
        except Exception as e:
            print(f"Final Exploitability: Error ({e})")
    
    avg_strategies = {}
    for infoset_key, node_data in info_sets.items():
        strat_sum = node_data['strategy_sum']
        total_sum = np.sum(strat_sum)
        if total_sum > 1e-9: # Avoid division by zero for infosets that might not have strategy sum
            avg_strategies[infoset_key] = strat_sum / total_sum
        else:
            # Fallback for infosets with no accumulated strategy
            avg_strategies[infoset_key] = np.zeros_like(strat_sum) # Default to zero strategy if no sum
            
    return avg_strategies, exploitability_history

def convert_mccfr_to_kuhn_strategies(info_sets_dict) -> tuple[KuhnStrategy, KuhnStrategy]:
    """Convert MCCFR info_sets format to KuhnStrategy format for exploitability calculation."""
    player1_strategy = KuhnStrategy()
    player2_strategy = KuhnStrategy()
    
    for infoset_key, node_data in info_sets_dict.items():
        card_name, history_tuple = infoset_key
        card = Card[card_name]  # Convert card name back to Card enum
        history_actions = tuple(Action[action_name] for action_name in history_tuple)
        
        # Determine which player this infoset belongs to based on history length
        # Player 0 acts first (empty history) and after even-length histories
        # Player 1 acts after odd-length histories
        is_player_0_turn = len(history_actions) % 2 == 0
        
        # Get average strategy for this infoset
        strat_sum = node_data['strategy_sum']
        total_sum = np.sum(strat_sum)
        
        if total_sum > 1e-9:  # Only process infosets with meaningful strategy
            avg_strategy = strat_sum / total_sum
            
            # Set probabilities for each action
            for action_idx, prob in enumerate(avg_strategy):
                if prob > 1e-9:  # Only set non-trivial probabilities
                    action = IDX_TO_ACTION[action_idx]
                    
                    if is_player_0_turn:
                        player1_strategy.set_action_probability(card, history_actions, action, prob)
                    else:
                        player2_strategy.set_action_probability(card, history_actions, action, prob)
    
    return player1_strategy, player2_strategy

def plot_classical_training_metrics(exploitability_history):
    """Plot exploitability during classical MCCFR training."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Plotting will be disabled.")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot exploitability
    if exploitability_history:
        # Create x-axis values (iterations where exploitability was calculated)
        exploitability_iterations = [100 + i * 100 for i in range(len(exploitability_history))]
        ax.plot(exploitability_iterations, exploitability_history, 'b-', linewidth=2, label='Exploitability', marker='o', markersize=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Exploitability')
        ax.set_title('Classical MCCFR: Exploitability Over Training')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add final value annotation
        if exploitability_history:
            final_exploit = exploitability_history[-1]
            ax.annotate(f'Final: {final_exploit:.6f}', 
                        xy=(exploitability_iterations[-1], final_exploit),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('classical_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'classical_training_metrics.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical MCCFR Training")
    parser.add_argument("--iterations", type=int, default=200000, help="Number of iterations for training")
    args = parser.parse_args()

    kuhn_game = KuhnGame()
    
    num_iterations = args.iterations
    print(f"Starting Classical MCCFR training for {num_iterations} iterations...")
    
    final_avg_strategies, exploitability_history = train_classical_mccfr(num_iterations, kuhn_game)

    # Print exploitability summary
    if exploitability_history:
        print(f"\n--- Classical MCCFR Exploitability Tracking Summary ---")
        print(f"Initial exploitability (iter 100): {exploitability_history[0]:.6f}")
        print(f"Final exploitability: {exploitability_history[-1]:.6f}")
        print(f"Best exploitability achieved: {min(exploitability_history):.6f}")
        print(f"Exploitability reduction: {((exploitability_history[0] - exploitability_history[-1]) / exploitability_history[0] * 100):.2f}%")

    # Plot training metrics
    if MATPLOTLIB_AVAILABLE:
        plot_classical_training_metrics(exploitability_history)
    else:
        print("\nNote: Plotting disabled due to matplotlib compatibility issues.")
        print("You can still access the training data from the returned variables.")

    print("\n--- Final Average Strategies (Sample) ---")
    count = 0
    for infoset, strategy_vector in final_avg_strategies.items():
        if count >= 20 and len(final_avg_strategies) > 20 : # Print a limited sample
             print("... (and more strategies)")
             break

        player_card_str = infoset[0]
        history_str = "".join(infoset[1]) if infoset[1] else "Root" # Infoset[1] is tuple of action names
        
        strat_display_list = []
        has_strategy = False
        for action_idx, prob in enumerate(strategy_vector):
            if prob > 1e-4: # Only show actions with non-trivial probability
                strat_display_list.append(f"{IDX_TO_ACTION[action_idx].name}: {prob:.3f}")
                has_strategy = True
        
        if has_strategy: # Only print infosets where a strategy was computed
            print(f"Infoset: Card {player_card_str}, Hist: '{history_str}' -> Strategy: {{{', '.join(strat_display_list)}}}")
            count += 1

    if count == 0 and len(final_avg_strategies) > 0:
        print("No strategies with significant probabilities found to display (all probabilities are too small or zero).")
    elif len(final_avg_strategies) == 0:
        print("No infosets were generated.")

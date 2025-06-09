import random
from typing import Dict, Tuple, List
from kuhn import KuhnGame, KuhnState, Card, Action
from collections import defaultdict
import itertools

class KuhnStrategy:
    """Represents a strategy for Kuhn poker as probabilities over actions for each information set."""
    
    def __init__(self):
        # Information sets are represented as (card, history_tuple)
        # Strategy maps (card, history) -> {action: probability}
        self.strategy = defaultdict(lambda: defaultdict(float))
        
    def get_action_probability(self, card: Card, history: tuple, action: Action) -> float:
        """Get probability of taking an action given card and history."""
        return self.strategy[(card, history)][action]
    
    def set_action_probability(self, card: Card, history: tuple, action: Action, prob: float):
        """Set probability of taking an action given card and history."""
        self.strategy[(card, history)][action] = prob
    
    def normalize_infoset(self, card: Card, history: tuple):
        """Normalize probabilities for an information set to sum to 1."""
        total = sum(self.strategy[(card, history)].values())
        if total > 0:
            for action in self.strategy[(card, history)]:
                self.strategy[(card, history)][action] /= total

def create_optimal_strategy() -> Tuple[KuhnStrategy, KuhnStrategy]:
    """Create the optimal Nash equilibrium strategies for both players."""
    
    # Player 1 strategy (first to act)
    player1_strategy = KuhnStrategy()
    
    # Player 1 initial actions (empty history)
    # K: Always bet
    player1_strategy.set_action_probability(Card.KING, (), Action.BET, 1.0)
    player1_strategy.set_action_probability(Card.KING, (), Action.CHECK, 0.0)
    
    # Q: Check, call a bet with probability ~1/3
    player1_strategy.set_action_probability(Card.QUEEN, (), Action.CHECK, 1.0)
    player1_strategy.set_action_probability(Card.QUEEN, (), Action.BET, 0.0)
    
    # J: Check, fold to a bet
    player1_strategy.set_action_probability(Card.JACK, (), Action.CHECK, 1.0)
    player1_strategy.set_action_probability(Card.JACK, (), Action.BET, 0.0)
    
    # Player 1 responses after checking and facing a bet
    # Q: Call with probability 1/3
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.CALL, 1.0/3.0)
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.FOLD, 2.0/3.0)
    
    # J: Always fold
    player1_strategy.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.FOLD, 1.0)
    player1_strategy.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.CALL, 0.0)
    
    # Player 2 strategy (second to act)
    player2_strategy = KuhnStrategy()
    
    # Player 2 responses to check
    # K: No specific action after check (would bet if given the chance)
    player2_strategy.set_action_probability(Card.KING, (Action.CHECK,), Action.CHECK, 0.0)
    player2_strategy.set_action_probability(Card.KING, (Action.CHECK,), Action.BET, 1.0)
    
    # Q: Check
    player2_strategy.set_action_probability(Card.QUEEN, (Action.CHECK,), Action.CHECK, 1.0)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.CHECK,), Action.BET, 0.0)
    
    # J: Always bluff (bet if checked to)
    player2_strategy.set_action_probability(Card.JACK, (Action.CHECK,), Action.BET, 1.0)
    player2_strategy.set_action_probability(Card.JACK, (Action.CHECK,), Action.CHECK, 0.0)
    
    # Player 2 responses to bet
    # K: Call a bet
    player2_strategy.set_action_probability(Card.KING, (Action.BET,), Action.CALL, 1.0)
    player2_strategy.set_action_probability(Card.KING, (Action.BET,), Action.FOLD, 0.0)
    
    # Q: Call a bet with probability 1/3
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.CALL, 1.0/3.0)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.FOLD, 2.0/3.0)
    
    # J: Fold to a bet
    player2_strategy.set_action_probability(Card.JACK, (Action.BET,), Action.FOLD, 1.0)
    player2_strategy.set_action_probability(Card.JACK, (Action.BET,), Action.CALL, 0.0)
    
    return player1_strategy, player2_strategy

def get_all_possible_hands() -> List[Tuple[Card, Card]]:
    """Get all possible card combinations for the two players."""
    hands = []
    cards = [Card.JACK, Card.QUEEN, Card.KING]
    for p1_card in cards:
        for p2_card in cards:
            if p1_card != p2_card:  # Players can't have the same card
                hands.append((p1_card, p2_card))
    return hands

def calculate_expected_value(player: int, player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy) -> float:
    """Calculate expected value for a player given both strategies."""
    
    total_ev = 0.0
    num_hands = 0
    
    # Iterate over all possible hands
    for p1_card, p2_card in get_all_possible_hands():
        ev = calculate_hand_expected_value(player, p1_card, p2_card, player1_strategy, player2_strategy)
        total_ev += ev
        num_hands += 1
    
    return total_ev / num_hands if num_hands > 0 else 0.0

def calculate_hand_expected_value(player: int, p1_card: Card, p2_card: Card, 
                                 player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy) -> float:
    """Calculate expected value for a specific hand."""
    
    game = KuhnGame()
    initial_state = KuhnState([p1_card, p2_card], 0, [1, 1], [])
    
    return calculate_state_expected_value(player, initial_state, player1_strategy, player2_strategy, 1.0)

def calculate_state_expected_value(player: int, state: KuhnState, 
                                  player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy, 
                                  reach_probability: float) -> float:
    """Recursively calculate expected value from a game state."""
    
    if state.is_terminal():
        returns = state.get_returns()
        return returns[player] * reach_probability
    
    current_player = state._current_player
    current_card = state._player_cards[current_player]
    history = tuple(state._history)
    
    # Get the appropriate strategy
    strategy = player1_strategy if current_player == 0 else player2_strategy
    
    legal_actions = state.get_legal_actions()
    expected_value = 0.0
    
    for action in legal_actions:
        action_prob = strategy.get_action_probability(current_card, history, action)
        
        if action_prob > 0:
            new_state = state.apply_action(action)
            new_reach_prob = reach_probability * action_prob
            
            ev = calculate_state_expected_value(player, new_state, player1_strategy, player2_strategy, new_reach_prob)
            expected_value += ev
    
    return expected_value

def find_best_response_value(player: int, opponent_strategy: KuhnStrategy, 
                           player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy) -> float:
    """Find the best response value for a player against an opponent's strategy."""
    
    # Create a best response strategy that always picks the best action
    best_response_strategy = KuhnStrategy()
    
    # Get all information sets for this player
    infosets = get_player_infosets(player)
    
    for card, history in infosets:
        best_action = None
        best_value = float('-inf')
        
        # Try each possible action and see which gives the highest expected value
        game = KuhnGame()
        # Create a dummy state to get legal actions
        dummy_cards = [Card.JACK, Card.QUEEN] if player == 0 else [Card.QUEEN, Card.JACK]
        dummy_state = KuhnState(dummy_cards, player, [1, 1], list(history))
        
        if not dummy_state.is_terminal():
            legal_actions = dummy_state.get_legal_actions()
            
            for action in legal_actions:
                # Temporarily set this action to probability 1.0
                temp_strategy = KuhnStrategy()
                temp_strategy.set_action_probability(card, history, action, 1.0)
                
                # Calculate EV with this action
                if player == 0:
                    ev = calculate_expected_value_with_infoset_override(0, temp_strategy, opponent_strategy, card, history)
                else:
                    ev = calculate_expected_value_with_infoset_override(1, player1_strategy, temp_strategy, card, history)
                
                if ev > best_value:
                    best_value = ev
                    best_action = action
        
        if best_action is not None:
            best_response_strategy.set_action_probability(card, history, best_action, 1.0)
    
    # Calculate expected value with best response strategy
    if player == 0:
        return calculate_expected_value(player, best_response_strategy, opponent_strategy)
    else:
        return calculate_expected_value(player, player1_strategy, best_response_strategy)

def get_player_infosets(player: int) -> List[Tuple[Card, tuple]]:
    """Get all possible information sets for a player."""
    infosets = []
    cards = [Card.JACK, Card.QUEEN, Card.KING]
    
    if player == 0:  # Player 1
        # Initial decision (empty history)
        for card in cards:
            infosets.append((card, ()))
        
        # Response after checking and facing a bet
        for card in cards:
            infosets.append((card, (Action.CHECK, Action.BET)))
            
    else:  # Player 2
        # Response to check
        for card in cards:
            infosets.append((card, (Action.CHECK,)))
        
        # Response to bet
        for card in cards:
            infosets.append((card, (Action.BET,)))
    
    return infosets

def calculate_expected_value_with_infoset_override(player: int, player1_strategy: KuhnStrategy, 
                                                 player2_strategy: KuhnStrategy, 
                                                 override_card: Card, override_history: tuple) -> float:
    """Calculate expected value, but only for hands where the player has the override card and faces the override history."""
    
    total_ev = 0.0
    num_relevant_hands = 0
    
    # Only consider hands where the specified player has the override card
    for p1_card, p2_card in get_all_possible_hands():
        if (player == 0 and p1_card == override_card) or (player == 1 and p2_card == override_card):
            # Check if this hand can reach the override history
            if can_reach_history(p1_card, p2_card, override_history, player1_strategy, player2_strategy):
                ev = calculate_hand_expected_value(player, p1_card, p2_card, player1_strategy, player2_strategy)
                total_ev += ev
                num_relevant_hands += 1
    
    return total_ev / num_relevant_hands if num_relevant_hands > 0 else 0.0

def can_reach_history(p1_card: Card, p2_card: Card, target_history: tuple, 
                     player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy) -> bool:
    """Check if a hand can reach a specific history with positive probability."""
    if len(target_history) == 0:
        return True
    
    # For simplicity, assume all histories are reachable with some probability
    # In a full implementation, you'd simulate the game tree
    return True

def calculate_exploitability(player1_strategy: KuhnStrategy, player2_strategy: KuhnStrategy) -> float:
    """Calculate the exploitability of the strategy profile."""
    
    # Get current expected values for both players
    p1_current_ev = calculate_expected_value(0, player1_strategy, player2_strategy)
    p2_current_ev = calculate_expected_value(1, player1_strategy, player2_strategy)
    
    # Get best response values
    p1_best_response_ev = find_best_response_value(0, player2_strategy, player1_strategy, player2_strategy)
    p2_best_response_ev = find_best_response_value(1, player1_strategy, player1_strategy, player2_strategy)
    
    # Exploitability is the sum of regrets
    p1_regret = p1_best_response_ev - p1_current_ev
    p2_regret = p2_best_response_ev - p2_current_ev
    
    exploitability = p1_regret + p2_regret
    
    return max(0.0, exploitability)  # Exploitability should be non-negative

def test_optimal_strategy_exploitability():
    """Test the exploitability of the optimal Nash equilibrium strategy."""
    
    print("Testing optimal strategy exploitability...")
    
    player1_strategy, player2_strategy = create_optimal_strategy()
    
    exploitability = calculate_exploitability(player1_strategy, player2_strategy)
    
    print(f"Exploitability of optimal strategy: {exploitability:.6f}")
    
    # Also print individual expected values for analysis
    p1_ev = calculate_expected_value(0, player1_strategy, player2_strategy)
    p2_ev = calculate_expected_value(1, player1_strategy, player2_strategy)
    
    print(f"Player 1 expected value: {p1_ev:.6f}")
    print(f"Player 2 expected value: {p2_ev:.6f}")
    print(f"Sum of expected values: {p1_ev + p2_ev:.6f}")
    
    return exploitability

def create_custom_strategy_example() -> Tuple[KuhnStrategy, KuhnStrategy]:
    """Create an example custom strategy that deviates from optimal play."""
    
    # Player 1 custom strategy (more aggressive)
    player1_strategy = KuhnStrategy()
    
    # Player 1 initial actions - make it more aggressive
    # K: Always bet (same as optimal)
    player1_strategy.set_action_probability(Card.KING, (), Action.BET, 1.0)
    player1_strategy.set_action_probability(Card.KING, (), Action.CHECK, 0.0)
    
    # Q: Bet more often instead of always checking
    player1_strategy.set_action_probability(Card.QUEEN, (), Action.CHECK, 0.6)
    player1_strategy.set_action_probability(Card.QUEEN, (), Action.BET, 0.4)
    
    # J: Sometimes bluff instead of always checking
    player1_strategy.set_action_probability(Card.JACK, (), Action.CHECK, 0.8)
    player1_strategy.set_action_probability(Card.JACK, (), Action.BET, 0.2)
    
    # Player 1 responses after checking and facing a bet
    # Q: Call more often (suboptimal)
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.CALL, 0.6)
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.FOLD, 0.4)
    
    # J: Sometimes call instead of always folding (very suboptimal)
    player1_strategy.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.FOLD, 0.9)
    player1_strategy.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.CALL, 0.1)
    
    # Player 2 custom strategy (more passive)
    player2_strategy = KuhnStrategy()
    
    # Player 2 responses to check - make it more passive
    # K: Sometimes check instead of always betting
    player2_strategy.set_action_probability(Card.KING, (Action.CHECK,), Action.CHECK, 0.3)
    player2_strategy.set_action_probability(Card.KING, (Action.CHECK,), Action.BET, 0.7)
    
    # Q: Always check (same as optimal)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.CHECK,), Action.CHECK, 1.0)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.CHECK,), Action.BET, 0.0)
    
    # J: Bluff less often
    player2_strategy.set_action_probability(Card.JACK, (Action.CHECK,), Action.BET, 0.5)
    player2_strategy.set_action_probability(Card.JACK, (Action.CHECK,), Action.CHECK, 0.5)
    
    # Player 2 responses to bet
    # K: Always call (same as optimal)
    player2_strategy.set_action_probability(Card.KING, (Action.BET,), Action.CALL, 1.0)
    player2_strategy.set_action_probability(Card.KING, (Action.BET,), Action.FOLD, 0.0)
    
    # Q: Call less often (more passive)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.CALL, 0.2)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.FOLD, 0.8)
    
    # J: Always fold (same as optimal)
    player2_strategy.set_action_probability(Card.JACK, (Action.BET,), Action.FOLD, 1.0)
    player2_strategy.set_action_probability(Card.JACK, (Action.BET,), Action.CALL, 0.0)
    
    return player1_strategy, player2_strategy

def create_always_aggressive_strategy() -> Tuple[KuhnStrategy, KuhnStrategy]:
    """Create a strategy where both players are overly aggressive."""
    
    # Player 1 always bets when possible
    player1_strategy = KuhnStrategy()
    
    # Always bet initially
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player1_strategy.set_action_probability(card, (), Action.BET, 1.0)
        player1_strategy.set_action_probability(card, (), Action.CHECK, 0.0)
    
    # Always call when facing a bet after checking
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player1_strategy.set_action_probability(card, (Action.CHECK, Action.BET), Action.CALL, 1.0)
        player1_strategy.set_action_probability(card, (Action.CHECK, Action.BET), Action.FOLD, 0.0)
    
    # Player 2 always bets/calls when possible
    player2_strategy = KuhnStrategy()
    
    # Always bet when checked to
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player2_strategy.set_action_probability(card, (Action.CHECK,), Action.BET, 1.0)
        player2_strategy.set_action_probability(card, (Action.CHECK,), Action.CHECK, 0.0)
    
    # Always call when facing a bet
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player2_strategy.set_action_probability(card, (Action.BET,), Action.CALL, 1.0)
        player2_strategy.set_action_probability(card, (Action.BET,), Action.FOLD, 0.0)
    
    return player1_strategy, player2_strategy

def create_always_passive_strategy() -> Tuple[KuhnStrategy, KuhnStrategy]:
    """Create a strategy where both players are overly passive."""
    
    # Player 1 always checks/folds
    player1_strategy = KuhnStrategy()
    
    # Always check initially
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player1_strategy.set_action_probability(card, (), Action.CHECK, 1.0)
        player1_strategy.set_action_probability(card, (), Action.BET, 0.0)
    
    # Always fold when facing a bet
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player1_strategy.set_action_probability(card, (Action.CHECK, Action.BET), Action.FOLD, 1.0)
        player1_strategy.set_action_probability(card, (Action.CHECK, Action.BET), Action.CALL, 0.0)
    
    # Player 2 always checks/folds
    player2_strategy = KuhnStrategy()
    
    # Always check when checked to
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player2_strategy.set_action_probability(card, (Action.CHECK,), Action.CHECK, 1.0)
        player2_strategy.set_action_probability(card, (Action.CHECK,), Action.BET, 0.0)
    
    # Always fold when facing a bet
    for card in [Card.KING, Card.QUEEN, Card.JACK]:
        player2_strategy.set_action_probability(card, (Action.BET,), Action.FOLD, 1.0)
        player2_strategy.set_action_probability(card, (Action.BET,), Action.CALL, 0.0)
    
    return player1_strategy, player2_strategy

def test_strategy_against_optimal(custom_strategy_name: str, custom_p1_strategy: KuhnStrategy, custom_p2_strategy: KuhnStrategy):
    """Test a custom strategy against the optimal strategy."""
    
    print(f"\n=== Testing {custom_strategy_name} ===")
    
    # Get optimal strategies
    optimal_p1, optimal_p2 = create_optimal_strategy()
    
    # Test custom strategy exploitability
    custom_exploitability = calculate_exploitability(custom_p1_strategy, custom_p2_strategy)
    print(f"Exploitability of {custom_strategy_name}: {custom_exploitability:.6f}")
    
    # Test custom vs optimal combinations
    # Custom P1 vs Optimal P2
    custom_p1_vs_optimal_p2_ev = calculate_expected_value(0, custom_p1_strategy, optimal_p2)
    optimal_p1_vs_optimal_p2_ev = calculate_expected_value(0, optimal_p1, optimal_p2)
    p1_loss = optimal_p1_vs_optimal_p2_ev - custom_p1_vs_optimal_p2_ev
    
    # Optimal P1 vs Custom P2
    optimal_p1_vs_custom_p2_ev = calculate_expected_value(1, optimal_p1, custom_p2_strategy)
    optimal_p1_vs_optimal_p2_p2_ev = calculate_expected_value(1, optimal_p1, optimal_p2)
    p2_loss = optimal_p1_vs_optimal_p2_p2_ev - optimal_p1_vs_custom_p2_ev
    
    print(f"Player 1 loss when using custom strategy vs optimal P2: {p1_loss:.6f}")
    print(f"Player 2 loss when using custom strategy vs optimal P1: {p2_loss:.6f}")
    
    # Expected values
    custom_p1_ev = calculate_expected_value(0, custom_p1_strategy, custom_p2_strategy)
    custom_p2_ev = calculate_expected_value(1, custom_p1_strategy, custom_p2_strategy)
    
    print(f"Custom P1 expected value: {custom_p1_ev:.6f}")
    print(f"Custom P2 expected value: {custom_p2_ev:.6f}")
    print(f"Sum of expected values: {(custom_p1_ev + custom_p2_ev):.6f}")
    
    return custom_exploitability

def compare_multiple_strategies():
    """Compare multiple strategies against the optimal one."""
    
    print("=== Kuhn Poker Strategy Comparison ===")
    
    # Test optimal strategy first
    optimal_p1, optimal_p2 = create_optimal_strategy()
    optimal_exploitability = calculate_exploitability(optimal_p1, optimal_p2)
    print(f"Optimal strategy exploitability: {optimal_exploitability:.6f}")
    
    # Test custom example strategy
    custom_p1, custom_p2 = create_custom_strategy_example()
    test_strategy_against_optimal("Custom Example Strategy", custom_p1, custom_p2)
    
    # Test always aggressive strategy
    aggressive_p1, aggressive_p2 = create_always_aggressive_strategy()
    test_strategy_against_optimal("Always Aggressive Strategy", aggressive_p1, aggressive_p2)
    
    # Test always passive strategy
    passive_p1, passive_p2 = create_always_passive_strategy()
    test_strategy_against_optimal("Always Passive Strategy", passive_p1, passive_p2)

def create_custom_strategy_interactive():
    """Helper function to create a custom strategy with user input."""
    print("\nCreating custom strategy...")
    print("This is a template function. You can modify the probabilities below.")
    
    # Create a template strategy similar to optimal but with some modifications
    player1_strategy = KuhnStrategy()
    player2_strategy = KuhnStrategy()
    
    # You can modify these probabilities to test different strategies
    # Player 1 strategy
    player1_strategy.set_action_probability(Card.KING, (), Action.BET, 1.0)  # Modify this
    player1_strategy.set_action_probability(Card.QUEEN, (), Action.CHECK, 1.0)  # Modify this
    player1_strategy.set_action_probability(Card.JACK, (), Action.CHECK, 1.0)  # Modify this
    
    # Player 1 responses
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.CALL, 0.5)  # Modify this
    player1_strategy.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.FOLD, 0.5)
    player1_strategy.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.FOLD, 1.0)
    
    # Player 2 strategy
    player2_strategy.set_action_probability(Card.KING, (Action.CHECK,), Action.BET, 1.0)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.CHECK,), Action.CHECK, 1.0)
    player2_strategy.set_action_probability(Card.JACK, (Action.CHECK,), Action.BET, 1.0)
    
    player2_strategy.set_action_probability(Card.KING, (Action.BET,), Action.CALL, 1.0)
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.CALL, 0.5)  # Modify this
    player2_strategy.set_action_probability(Card.QUEEN, (Action.BET,), Action.FOLD, 0.5)
    player2_strategy.set_action_probability(Card.JACK, (Action.BET,), Action.FOLD, 1.0)
    
    return player1_strategy, player2_strategy

if __name__ == "__main__":
    # Test optimal strategy exploitability first
    print("=== Testing Optimal Strategy ===")
    test_optimal_strategy_exploitability()
    
    # Compare multiple strategies against optimal
    print("\n" + "="*50)
    compare_multiple_strategies()
    
    print("\n" + "="*50)
    print("=== How to test your own custom strategy ===")
    print("1. Create your strategy using the KuhnStrategy class")
    print("2. Set probabilities for each (card, history) combination")
    print("3. Call test_strategy_against_optimal() to see how exploitable it is")
    print("\nExample:")
    print("my_p1, my_p2 = create_custom_strategy_interactive()")
    print("test_strategy_against_optimal('My Strategy', my_p1, my_p2)")
    
    # Example of how to create and test a simple custom strategy
    print("\n=== Example: Testing a Simple Custom Strategy ===")
    
    # Create a simple strategy where Player 1 always bets with Kings and Queens
    simple_p1 = KuhnStrategy()
    simple_p2 = KuhnStrategy()
    
    # Player 1: Bet with K and Q, check with J
    simple_p1.set_action_probability(Card.KING, (), Action.BET, 1.0)
    simple_p1.set_action_probability(Card.QUEEN, (), Action.BET, 1.0)  # Different from optimal
    simple_p1.set_action_probability(Card.JACK, (), Action.CHECK, 1.0)
    
    # Player 1 responses (same as optimal for simplicity)
    simple_p1.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.CALL, 1.0/3.0)
    simple_p1.set_action_probability(Card.QUEEN, (Action.CHECK, Action.BET), Action.FOLD, 2.0/3.0)
    simple_p1.set_action_probability(Card.JACK, (Action.CHECK, Action.BET), Action.FOLD, 1.0)
    
    # Player 2: Use optimal strategy
    optimal_p1, optimal_p2 = create_optimal_strategy()
    simple_p2 = optimal_p2  # Use optimal strategy for Player 2
    
    test_strategy_against_optimal("Simple Custom Strategy (P1 bets with K and Q)", simple_p1, simple_p2)

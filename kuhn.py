import random
from enum import Enum

# Enums for card ranks and player actions
class Card(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3

def card_to_string(card: Card) -> str:
    if card == Card.JACK: return "J"
    if card == Card.QUEEN: return "Q"
    if card == Card.KING: return "K"
    return "Unknown"


class KuhnState:
    def __init__(self, player_cards: list[Card], current_player: int, bets: list[int], history: list[Action]):
        self._player_cards = player_cards
        self._current_player = current_player
        self._bets = bets
        self._history = history

    def is_terminal(self) -> bool:
        if not self._history: # Equivalent to .empty() check
            return False

        last_action = self._history[-1]
        if last_action == Action.FOLD:
            return True

        # Terminal cases for Kuhn poker:
        # 1. CHECK, CHECK (both players check)
        # 2. BET, CALL (one player bets, other calls)
        # 3. CHECK, BET, CALL (check, bet, call sequence)
        # 4. BET, FOLD (bet and fold, handled above)
        # 5. CHECK, BET, FOLD (check, bet, fold, handled above)
        
        if len(self._history) == 2:
            # Cases: CHECK, CHECK or BET, CALL
            if self._history[0] == Action.CHECK and self._history[1] == Action.CHECK:
                return True
            if self._history[0] == Action.BET and self._history[1] == Action.CALL:
                return True
        elif len(self._history) == 3:
            # Case: CHECK, BET, CALL
            if (self._history[0] == Action.CHECK and 
                self._history[1] == Action.BET and 
                self._history[2] == Action.CALL):
                return True
        return False

    def get_legal_actions(self) -> list[Action]:
        legal_actions = []
        if not self._history:
            legal_actions.append(Action.CHECK)
            legal_actions.append(Action.BET)
        elif len(self._history) == 1:
            prev_action = self._history[0]
            if prev_action == Action.CHECK:
                legal_actions.append(Action.CHECK)
                legal_actions.append(Action.BET)
            elif prev_action == Action.BET:
                legal_actions.append(Action.FOLD)
                legal_actions.append(Action.CALL)
        elif len(self._history) == 2:
            # This case handles when P0 bets, P1 bets, and it's P0's turn again
            prev_action = self._history[1]
            if prev_action == Action.BET:
                legal_actions.append(Action.FOLD)
                legal_actions.append(Action.CALL)
        return legal_actions

    def apply_action(self, action: Action):
        new_history = list(self._history) # Create a copy
        new_history.append(action)

        new_bets = list(self._bets) # Create a copy
        new_current_player = 1 - self._current_player

        if action == Action.BET:
            new_bets[self._current_player] += 1 # Kuhn bet size is 1
        elif action == Action.CALL:
            # Player calls, matches the opponent's bet.
            # The opponent's bet is new_bets[1 - self._current_player] - new_bets[self._current_player]
            new_bets[self._current_player] = new_bets[1 - self._current_player]
        return KuhnState(self._player_cards, new_current_player, new_bets, new_history)

    def get_returns(self) -> list[float]:
        if not self.is_terminal():
            return [] # Should only be called on terminal states

        returns = [0.0, 0.0] # Assuming 2 players based on the C++ code

        if self._history[-1] == Action.FOLD:
            winner = self._current_player # The player who didn't fold
            loser = 1 - self._current_player
            returns[winner] = float(self._bets[loser]) # Winner gets loser's bet
            returns[loser] = -float(self._bets[loser]) # Loser loses their bet
        else:
            # Showdown: CHECK, CHECK or BET, CALL
            winning_player = 0 if self._player_cards[0].value > self._player_cards[1].value else 1
            losing_player = 1 - winning_player
            
            pot = float(self._bets[0] + self._bets[1])
            returns[winning_player] = pot - self._bets[winning_player]
            returns[losing_player] = -float(self._bets[losing_player])
        return returns

    def __str__(self) -> str:
        s = f"Cards: [{card_to_string(self._player_cards[0])}, {card_to_string(self._player_cards[1])}] "
        s += f"Current Player: {self._current_player} "
        s += f"Bets: [{self._bets[0]}, {self._bets[1]}] "
        s += "History: "
        for a in self._history:
            if a == Action.FOLD: s += "F "
            elif a == Action.CHECK: s += "X "
            elif a == Action.CALL: s += "C "
            elif a == Action.BET: s += "B "
        return s.strip()


class KuhnGame:
    def __init__(self):
        self._num_players = 2
        self._deck = [Card.JACK, Card.QUEEN, Card.KING]
        self._initial_pot_size = 2
        self._bet_size = 1
        self._raises_allowed = 1 # Kuhn poker has at most 1 raise

    def get_initial_state(self):
        # Shuffle and deal cards
        shuffled_deck = list(self._deck) # Create a mutable copy
        random.shuffle(shuffled_deck)

        player_cards = [shuffled_deck[0], shuffled_deck[1]]
        bets = [1, 1] # Each player posts an ante of 1
        history = []

        return KuhnState(player_cards, 0, bets, history) # Player 0 starts

    def get_num_players(self) -> int:
        return self._num_players


if __name__ == "__main__":
    game = KuhnGame()
    state = game.get_initial_state()

    print(f"Initial State: {state}")

    # Example game play (simplified)
    while not state.is_terminal():
        legal_actions = state.get_legal_actions()
        if not legal_actions: # Check if empty
            break

        # For demonstration, just pick the first legal action
        chosen_action = legal_actions[0]
        print(f"Player {state._current_player} chooses action: ", end="")
        if chosen_action == Action.FOLD: print("FOLD")
        elif chosen_action == Action.CHECK: print("CHECK")
        elif chosen_action == Action.CALL: print("CALL")
        elif chosen_action == Action.BET: print("BET")

        state = state.apply_action(chosen_action)
        print(f"New State: {state}")

    if state.is_terminal():
        returns = state.get_returns()
        print(f"Terminal State. Returns: [{returns[0]}, {returns[1]}]") 
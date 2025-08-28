from enum import Enum
import random
import collections

class Card(Enum):
    J_HEARTS = 0
    Q_HEARTS = 1
    K_HEARTS = 2
    J_SPADES = 3
    Q_SPADES = 4
    K_SPADES = 5

def get_rank_value(card: Card) -> int:
    if card in [Card.J_HEARTS, Card.J_SPADES]: return 0
    if card in [Card.Q_HEARTS, Card.Q_SPADES]: return 1
    if card in [Card.K_HEARTS, Card.K_SPADES]: return 2
    return -1 # Should not happen

def card_to_string(card: Card) -> str:
    if card == Card.J_HEARTS: return "J♥"
    if card == Card.Q_HEARTS: return "Q♥"
    if card == Card.K_HEARTS: return "K♥"
    if card == Card.J_SPADES: return "J♠"
    if card == Card.Q_SPADES: return "Q♠"
    if card == Card.K_SPADES: return "K♠"
    return "Unknown"

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4 

class LeducState:
    def __init__(self, player_cards: list[Card], board_cards: list[Card], current_player: int, bets: list[int], history: list[Action], street: int):
        self._player_cards = player_cards
        self._board_cards = board_cards
        self._current_player = current_player
        self._bets = bets
        self._history = history
        self._street = street

    def is_terminal(self) -> bool:
        if not self._history: # Equivalent to _history.empty()
            return False

        last_action = self._history[-1]

        # 1. Fold: Game ends immediately
        if last_action == Action.FOLD:
            return True

        # 2. Showdown after betting round completes.
        # A betting round completes if:
        # a) Two checks occur on the same street (CHECK, CHECK)
        # b) A bet/raise is called (BET, CALL) or (RAISE, CALL)

        if len(self._history) >= 2:
            # If the last action was CALL
            if last_action == Action.CALL:
                # Check previous action to see if it was a BET or RAISE
                if self._history[-2] in [Action.BET, Action.RAISE]:
                    # If it's the flop street, and a call happened, it's terminal.
                    return (self._street == 1) # Only terminal if on flop street
            # If two checks happened
            if self._history[-2] == Action.CHECK and last_action == Action.CHECK:
                return (self._street == 1) # Only terminal if on flop street

        return False 

    def get_legal_actions(self) -> list[Action]:
        legal_actions = []
        raises_on_street = 0
        for a in self._history:
            if a in [Action.BET, Action.RAISE]:
                raises_on_street += 1

        current_player_bet = self._bets[self._current_player]
        opponent_player_bet = self._bets[1 - self._current_player]
        facing_bet_or_raise = (current_player_bet < opponent_player_bet)

        if facing_bet_or_raise:
            legal_actions.append(Action.FOLD)
            legal_actions.append(Action.CALL)
            # Can raise if max_raises (2) not reached
            if raises_on_street < 2:
                legal_actions.append(Action.RAISE)
        else: # Not facing a bet (bets are equal, or it's the first action)
            legal_actions.append(Action.CHECK)
            # Can bet if max_raises (2) not reached
            if raises_on_street < 2:
                 legal_actions.append(Action.BET)

        return legal_actions

    def apply_action(self, action: Action):
        new_history = list(self._history) # Create a copy
        new_history.append(action)

        new_bets = list(self._bets) # Create a copy
        new_current_player = 1 - self._current_player # Switch player after action

        # Determine the current bet size for the street
        current_bet_increment = 1 if self._street == 0 else 2 # Small bet pre-flop, big bet flop

        if action == Action.BET:
            new_bets[self._current_player] += current_bet_increment
        elif action == Action.RAISE:
            # A raise means matching opponent's current bet and then adding the bet increment.
            amount_to_call = new_bets[1 - self._current_player] - new_bets[self._current_player]
            new_bets[self._current_player] += amount_to_call + current_bet_increment # Match + Raise amount
        elif action == Action.CALL:
            new_bets[self._current_player] = new_bets[1 - self._current_player]

        return LeducState(self._player_cards, self._board_cards, new_current_player, new_bets, new_history, self._street) 

    def get_returns(self) -> list[float]:
        if not self.is_terminal():
            return [] # Should only be called on terminal states

        returns = [0.0] * len(self._player_cards)

        # Case 1: Fold
        if self._history[-1] == Action.FOLD:
            winner = self._current_player # The player who didn't fold is the winner
            loser = 1 - self._current_player
            returns[winner] = self._bets[loser]
            returns[loser] = -self._bets[loser]
        else:
            # Case 2: Showdown (CHECK, CHECK or BET, CALL or RAISE, CALL)

            p0_hand_card = self._player_cards[0]
            p1_hand_card = self._player_cards[1]
            board = self._board_cards[0] # Assumes board_cards has one element at showdown

            p0_has_pair = (p0_hand_card == board)
            p1_has_pair = (p1_hand_card == board)

            winning_player = -1 # -1 indicates a tie

            if p0_has_pair and p1_has_pair:
                # Both have pairs, higher rank wins the pair
                if get_rank_value(p0_hand_card) > get_rank_value(p1_hand_card):
                    winning_player = 0
                elif get_rank_value(p1_hand_card) > get_rank_value(p0_hand_card):
                    winning_player = 1
                # else it's a tie (winning_player remains -1)
            elif p0_has_pair:
                winning_player = 0
            elif p1_has_pair:
                winning_player = 1
            else:
                # No pairs, higher hole card wins
                if get_rank_value(p0_hand_card) > get_rank_value(p1_hand_card):
                    winning_player = 0
                elif get_rank_value(p1_hand_card) > get_rank_value(p0_hand_card):
                    winning_player = 1
                # else it's a tie (winning_player remains -1)

            pot = self._bets[0] + self._bets[1]
            if winning_player != -1:
                losing_player = 1 - winning_player
                returns[winning_player] = pot - self._bets[winning_player]
                returns[losing_player] = -self._bets[losing_player]
            else:
                # Split pot (for ties, usually 0 return for each as they get their money back)
                returns[0] = 0.0
                returns[1] = 0.0
        return returns

    def to_string(self) -> str:
        s = f"P0 Card: {card_to_string(self._player_cards[0])}, P1 Card: {card_to_string(self._player_cards[1])} "
        s += "Board Card: "
        if self._board_cards:
            s += card_to_string(self._board_cards[0])
        else:
            s += "None"
        s += f" Current Player: {self._current_player} "
        s += f"Bets: [{self._bets[0]}, {self._bets[1]}] "
        s += f"Street: {self._street} "
        s += "History: "
        for a in self._history:
            if a == Action.FOLD: s += "F "
            elif a == Action.CHECK: s += "X "
            elif a == Action.CALL: s += "C "
            elif a == Action.BET: s += "B "
            elif a == Action.RAISE: s += "R "
        return s 

class LeducGame:
    def __init__(self):
        self._num_players = 2
        self._deck = [Card.J_HEARTS, Card.Q_HEARTS, Card.K_HEARTS, Card.J_SPADES, Card.Q_SPADES, Card.K_SPADES]
        self._initial_pot_size = 2
        self._small_bet_size = 1
        self._big_bet_size = 2
        self._max_raises = 2

    def get_initial_state(self):
        # Shuffle and deal cards
        shuffled_deck = list(self._deck) # Create a copy
        random.shuffle(shuffled_deck)

        player_cards = [shuffled_deck[0], shuffled_deck[1]] # Each player gets one hole card
        board_cards = [] # No community cards initially
        bets = [1, 1] # Each player posts an ante of 1
        history = []

        return LeducState(player_cards, board_cards, 0, bets, history, 0) # Player 0 starts, pre-flop street

    def get_num_players(self) -> int:
        return self._num_players

    # Method to deal community card (flop)
    def deal_flop(self, state: LeducState) -> LeducState:
        remaining_deck_for_flop = []
        current_full_deck = list(self._deck)

        # Create a frequency map to keep track of cards dealt
        card_counts = collections.Counter(current_full_deck)

        # Decrement counts for player cards
        for p_card in state._player_cards:
            card_counts[p_card] -= 1

        # Populate remaining_deck_for_flop with available cards
        for card_rank, count in card_counts.items():
            for _ in range(count):
                remaining_deck_for_flop.append(card_rank)

        # Shuffle the remaining deck and pick one for the board.
        random.shuffle(remaining_deck_for_flop)

        new_board_cards = [remaining_deck_for_flop[0]]

        # Reset history for the new street, current player starts new street.
        return LeducState(state._player_cards, new_board_cards, 0, state._bets, [], 1) # Player 0 starts on flop, history cleared. 

if __name__ == "__main__":
    game = LeducGame()
    state = game.get_initial_state()

    print(f"Initial State: {state.to_string()}")

    # Example game play
    while not state.is_terminal():
        legal_actions = state.get_legal_actions()
        
        # Check if betting round is complete and we need to deal the flop
        if state._street == 0 and len(state._history) >= 2:
            last_action = state._history[-1]
            prev_action = state._history[-2]
            
            # If betting round complete (CHECK-CHECK or BET/RAISE-CALL), deal flop
            if ((last_action == Action.CHECK and prev_action == Action.CHECK) or
                (last_action == Action.CALL and (prev_action == Action.BET or prev_action == Action.RAISE))):
                state = game.deal_flop(state)
                print(f"Dealt flop. New State: {state.to_string()}")
                continue

        if not legal_actions:
            print("No legal actions available. This should not happen.")
            break

        # For demonstration, just pick the first legal action
        chosen_action = legal_actions[0]
        action_str = ""
        if chosen_action == Action.FOLD: action_str = "FOLD"
        elif chosen_action == Action.CHECK: action_str = "CHECK"
        elif chosen_action == Action.CALL: action_str = "CALL"
        elif chosen_action == Action.BET: action_str = "BET"
        elif chosen_action == Action.RAISE: action_str = "RAISE"
        print(f"Player {state._current_player} chooses action: {action_str}")

        state = state.apply_action(chosen_action)
        print(f"New State: {state.to_string()}")

    if state.is_terminal():
        returns = state.get_returns()
        print(f"Terminal State. Returns: [{returns[0]}, {returns[1]}]") 
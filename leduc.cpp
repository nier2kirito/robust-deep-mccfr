#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include <random>
#include <chrono>


// Enum for card ranks
enum Card {
    J_HEARTS,
    Q_HEARTS,
    K_HEARTS,
    J_SPADES,
    Q_SPADES,
    K_SPADES
};

// Helper to get rank of a card for comparison
int get_rank_value(Card card) {
    if (card == J_HEARTS || card == J_SPADES) return 0;
    if (card == Q_HEARTS || card == Q_SPADES) return 1;
    if (card == K_HEARTS || card == K_SPADES) return 2;
    return -1; // Should not happen
}

std::string card_to_string(Card card) {
    switch (card) {
        case J_HEARTS: return "J♥";
        case Q_HEARTS: return "Q♥";
        case K_HEARTS: return "K♥";
        case J_SPADES: return "J♠";
        case Q_SPADES: return "Q♠";
        case K_SPADES: return "K♠";
    }
    return "Unknown";
}

// Enum for player actions
enum Action {
    FOLD, CHECK, CALL, BET, RAISE
};

// Represents the state of a Leduc Poker game
class LeducState {
public:
    std::vector<Card> _player_cards; // Cards dealt to each player (hole card)
    std::vector<Card> _board_cards;  // Community cards
    int _current_player;             // Current player to act (0 or 1)
    std::vector<int> _bets;          // Bets made by each player
    std::vector<Action> _history;    // History of actions taken in the current street
    int _street;                     // Current street (0: pre-flop, 1: flop)

    LeducState(std::vector<Card> player_cards, std::vector<Card> board_cards, int current_player, std::vector<int> bets, std::vector<Action> history, int street)
        : _player_cards(player_cards), _board_cards(board_cards), _current_player(current_player), _bets(bets), _history(history), _street(street) {}

    bool is_terminal() const {
        if (_history.empty()) return false;

        Action last_action = _history.back();

        // 1. Fold: Game ends immediately
        if (last_action == FOLD) return true;

        // 2. Showdown after betting round completes.
        // A betting round completes if:
        // a) Two checks occur on the same street (CHECK, CHECK)
        // b) A bet/raise is called (BET, CALL) or (RAISE, CALL)

        if (_history.size() >= 2) {
            // If the last action was CALL
            if (last_action == CALL) {
                // Check previous action to see if it was a BET or RAISE
                if (_history[_history.size() - 2] == BET || _history[_history.size() - 2] == RAISE) {
                    // If it's the flop street, and a call happened, it's terminal.
                    return (_street == 1); // Only terminal if on flop street
                }
            }
            // If two checks happened
            if (_history[_history.size() - 2] == CHECK && last_action == CHECK) {
                return (_street == 1); // Only terminal if on flop street
            }
        }

        return false;
    }

    std::vector<Action> get_legal_actions() const {
        std::vector<Action> legal_actions;
        int raises_on_street = 0;
        for (Action a : _history) {
            if (a == BET || a == RAISE) {
                raises_on_street++;
            }
        }

        int current_player_bet = _bets[_current_player];
        int opponent_player_bet = _bets[1 - _current_player];
        bool facing_bet_or_raise = (current_player_bet < opponent_player_bet);

        if (facing_bet_or_raise) {
            legal_actions.push_back(FOLD);
            legal_actions.push_back(CALL);
            // Can raise if max_raises (2) not reached
            if (raises_on_street < 2) { 
                legal_actions.push_back(RAISE);
            }
        } else { // Not facing a bet (bets are equal, or it's the first action)
            legal_actions.push_back(CHECK);
            // Can bet if max_raises (2) not reached
            if (raises_on_street < 2) { 
                 legal_actions.push_back(BET);
            }
        }

        return legal_actions;
    }

    LeducState apply_action(Action action) const {
        std::vector<Action> new_history = _history;
        new_history.push_back(action);

        std::vector<int> new_bets = _bets;
        int new_current_player = 1 - _current_player; // Switch player after action

        // Determine the current bet size for the street
        int current_bet_increment = (_street == 0) ? 1 : 2; // Small bet pre-flop, big bet flop

        if (action == BET) {
            new_bets[_current_player] += current_bet_increment;
        } else if (action == RAISE) {
            // A raise means matching opponent's current bet and then adding the bet increment.
            int amount_to_call = new_bets[1 - _current_player] - new_bets[_current_player];
            new_bets[_current_player] += amount_to_call + current_bet_increment; // Match + Raise amount
        } else if (action == CALL) {
            new_bets[_current_player] = new_bets[1 - _current_player];
        }

        return LeducState(_player_cards, _board_cards, new_current_player, new_bets, new_history, _street);
    }

    std::vector<double> get_returns() const {
        if (!is_terminal()) {
            return {}; // Should only be called on terminal states
        }

        std::vector<double> returns(_player_cards.size());

        // Case 1: Fold
        if (_history.back() == FOLD) {
            int winner = _current_player; // The player who didn't fold is the winner
            int loser = 1 - _current_player;
            returns[winner] = _bets[loser]; 
            returns[loser] = -_bets[loser]; 
        } else {
            // Case 2: Showdown (CHECK, CHECK or BET, CALL or RAISE, CALL)

            Card p0_hand_card = _player_cards[0];
            Card p1_hand_card = _player_cards[1];
            Card board = _board_cards[0]; // Assumes board_cards has one element at showdown

            bool p0_has_pair = (p0_hand_card == board);
            bool p1_has_pair = (p1_hand_card == board);

            int winning_player = -1; // -1 indicates a tie

            if (p0_has_pair && p1_has_pair) {
                // Both have pairs, higher rank wins the pair
                if (get_rank_value(p0_hand_card) > get_rank_value(p1_hand_card)) {
                    winning_player = 0;
                } else if (get_rank_value(p1_hand_card) > get_rank_value(p0_hand_card)) {
                    winning_player = 1;
                }
                // else it's a tie (winning_player remains -1)
            } else if (p0_has_pair) {
                winning_player = 0;
            } else if (p1_has_pair) {
                winning_player = 1;
            } else {
                // No pairs, higher hole card wins
                if (get_rank_value(p0_hand_card) > get_rank_value(p1_hand_card)) {
                    winning_player = 0;
                } else if (get_rank_value(p1_hand_card) > get_rank_value(p0_hand_card)) {
                    winning_player = 1;
                }
                // else it's a tie (winning_player remains -1)
            }

            double pot = _bets[0] + _bets[1];
            if (winning_player != -1) {
                int losing_player = 1 - winning_player;
                returns[winning_player] = pot - _bets[winning_player];
                returns[losing_player] = -_bets[losing_player];
            } else {
                // Split pot (for ties, usually 0 return for each as they get their money back)
                returns[0] = 0.0;
                returns[1] = 0.0;
            }
        }
        return returns;
    }

    std::string to_string() const {
        std::string s = "P0 Card: " + card_to_string(_player_cards[0]) + ", P1 Card: " + card_to_string(_player_cards[1]) + " ";
        s += "Board Card: ";
        if (!_board_cards.empty()) {
            s += card_to_string(_board_cards[0]);
        } else {
            s += "None";
        }
        s += " Current Player: " + std::to_string(_current_player) + " ";
        s += "Bets: [" + std::to_string(_bets[0]) + ", " + std::to_string(_bets[1]) + "] ";
        s += "Street: " + std::to_string(_street) + " ";
        s += "History: ";
        for (Action a : _history) {
            if (a == FOLD) s += "F ";
            else if (a == CHECK) s += "X ";
            else if (a == CALL) s += "C ";
            else if (a == BET) s += "B ";
            else if (a == RAISE) s += "R ";
        }
        return s;
    }
};

// Represents the Leduc Poker game
class LeducGame {
public:
    int _num_players;
    std::array<Card, 6> _deck; // Two suits of Jack, Queen, King (e.g., Jc, Qc, Kc, Js, Qs, Ks)
    int _initial_pot_size;
    int _small_bet_size;
    int _big_bet_size;
    int _max_raises; // Max raises per street

    LeducGame() : _num_players(2), _deck({J_HEARTS, Q_HEARTS, K_HEARTS, J_SPADES, Q_SPADES, K_SPADES}), _initial_pot_size(2), _small_bet_size(1), _big_bet_size(2), _max_raises(2) {}

    LeducState get_initial_state() const {
        // Shuffle and deal cards
        std::vector<Card> shuffled_deck(_deck.begin(), _deck.end());
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng(seed);
        std::shuffle(shuffled_deck.begin(), shuffled_deck.end(), rng);

        std::vector<Card> player_cards = {shuffled_deck[0], shuffled_deck[1]}; // Each player gets one hole card
        std::vector<Card> board_cards; // No community cards initially
        std::vector<int> bets = {1, 1}; // Each player posts an ante of 1
        std::vector<Action> history;

        return LeducState(player_cards, board_cards, 0, bets, history, 0); // Player 0 starts, pre-flop street
    }

    int get_num_players() const {
        return _num_players;
    }

    // Method to deal community card (flop)
    LeducState deal_flop(const LeducState& state) const {
        std::vector<Card> remaining_deck_for_flop;
        std::vector<Card> current_full_deck(_deck.begin(), _deck.end());

        // Create a frequency map to keep track of cards dealt
        std::map<Card, int> card_counts;
        for(Card c : current_full_deck) {
            card_counts[c]++;
        }

        // Decrement counts for player cards
        for (Card p_card : state._player_cards) {
            card_counts[p_card]--;
        }

        // Populate remaining_deck_for_flop with available cards
        for (auto const& [card_rank, count] : card_counts) {
            for (int i = 0; i < count; ++i) {
                remaining_deck_for_flop.push_back(card_rank);
            }
        }

        // Shuffle the remaining deck and pick one for the board.
        // Note: std::random_shuffle is deprecated in C++17. For modern C++, use std::shuffle with a random engine.
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng(seed);
        std::shuffle(remaining_deck_for_flop.begin(), remaining_deck_for_flop.end(), rng);

        std::vector<Card> new_board_cards = {remaining_deck_for_flop[0]};

        // Reset history for the new street, current player starts new street.
        return LeducState(state._player_cards, new_board_cards, 0, state._bets, {}, 1); // Player 0 starts on flop, history cleared.
    }
};

#include <iostream>
#include <random>
#include <chrono>
int main() {
    LeducGame game;
    LeducState state = game.get_initial_state();

    std::cout << "Initial State: " << state.to_string() << std::endl;

    // Example game play
    while (!state.is_terminal()) {
        std::vector<Action> legal_actions = state.get_legal_actions();
        
        // Check if betting round is complete and we need to deal the flop
        if (state._street == 0 && state._history.size() >= 2) {
            Action last_action = state._history.back();
            Action prev_action = state._history[state._history.size() - 2];
            
            // If betting round complete (CHECK-CHECK or BET/RAISE-CALL), deal flop
            if ((last_action == CHECK && prev_action == CHECK) ||
                (last_action == CALL && (prev_action == BET || prev_action == RAISE))) {
                state = game.deal_flop(state);
                std::cout << "Dealt flop. New State: " << state.to_string() << std::endl;
                continue;
            }
        }

        if (legal_actions.empty()) {
            std::cout << "No legal actions available. This should not happen." << std::endl;
            break;
        }

        // For demonstration, just pick the first legal action
        Action chosen_action = legal_actions[0];
        std::cout << "Player " << state._current_player << " chooses action: ";
        if (chosen_action == FOLD) std::cout << "FOLD";
        else if (chosen_action == CHECK) std::cout << "CHECK";
        else if (chosen_action == CALL) std::cout << "CALL";
        else if (chosen_action == BET) std::cout << "BET";
        else if (chosen_action == RAISE) std::cout << "RAISE";
        std::cout << std::endl;

        state = state.apply_action(chosen_action);
        std::cout << "New State: " << state.to_string() << std::endl;
    }

    if (state.is_terminal()) {
        std::vector<double> returns = state.get_returns();
        std::cout << "Terminal State. Returns: [" << returns[0] << ", " << returns[1] << "]" << std::endl;
    }

    return 0;
}
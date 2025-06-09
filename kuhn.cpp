#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <map>
#include <iostream>
#include <random>
#include <chrono>

// Enum for card ranks
enum Card {
    JACK, QUEEN, KING
};

// Helper to get rank of a card for comparison
int get_rank_value(Card card) {
    if (card == JACK) return 0;
    if (card == QUEEN) return 1;
    if (card == KING) return 2;
    return -1; // Should not happen
}

std::string card_to_string(Card card) {
    switch (card) {
        case JACK: return "J";
        case QUEEN: return "Q";
        case KING: return "K";
    }
    return "Unknown";
}

// Enum for player actions
enum Action {
    FOLD, CHECK, CALL, BET
};

// Represents the state of a Kuhn Poker game
class KuhnState {
public:
    std::vector<Card> _player_cards; // Cards dealt to each player
    int _current_player;             // Current player to act (0 or 1)
    std::vector<int> _bets;          // Bets made by each player
    std::vector<Action> _history;    // History of actions taken in the current street

    KuhnState(std::vector<Card> player_cards, int current_player, std::vector<int> bets, std::vector<Action> history)
        : _player_cards(player_cards), _current_player(current_player), _bets(bets), _history(history) {}

    bool is_terminal() const {
        if (_history.empty()) return false;

        Action last_action = _history.back();
        if (last_action == FOLD) return true;

        if (_history.size() == 2) {
            // Cases: CHECK, CHECK or BET, CALL
            if (_history[0] == CHECK && _history[1] == CHECK) return true;
            if (_history[0] == BET && _history[1] == CALL) return true;
        }
        return false;
    }

    std::vector<Action> get_legal_actions() const {
        std::vector<Action> legal_actions;
        if (_history.empty()) {
            legal_actions.push_back(CHECK);
            legal_actions.push_back(BET);
        } else if (_history.size() == 1) {
            Action prev_action = _history[0];
            if (prev_action == CHECK) {
                legal_actions.push_back(CHECK);
                legal_actions.push_back(BET);
            } else if (prev_action == BET) {
                legal_actions.push_back(FOLD);
                legal_actions.push_back(CALL);
            }
        } else if (_history.size() == 2) {
            // This case handles when P0 bets, P1 bets, and it's P0's turn again
            Action prev_action = _history[1];
            if (prev_action == BET) {
                legal_actions.push_back(FOLD);
                legal_actions.push_back(CALL);
            }
        }
        return legal_actions;
    }

    KuhnState apply_action(Action action) const {
        std::vector<Action> new_history = _history;
        new_history.push_back(action);

        std::vector<int> new_bets = _bets;
        int new_current_player = 1 - _current_player;

        if (action == BET) {
            new_bets[_current_player] += 1; // Kuhn bet size is 1
        } else if (action == CALL) {
            // Player calls, matches the opponent's bet. The opponent's bet is _bets[1-_current_player] - _bets[_current_player]
            new_bets[_current_player] = new_bets[1 - _current_player];
        }
        return KuhnState(_player_cards, new_current_player, new_bets, new_history);
    }

    std::vector<double> get_returns() const {
        if (!is_terminal()) {
            return {}; // Should only be called on terminal states
        }

        std::vector<double> returns(_player_cards.size());

        if (_history.back() == FOLD) {
            int winner = _current_player; // The player who didn't fold
            int loser = 1 - _current_player;
            returns[winner] = _bets[loser]; // Winner gets loser's bet
            returns[loser] = -_bets[loser]; // Loser loses their bet
        } else {
            // Showdown: CHECK, CHECK or BET, CALL
            int winning_player = (_player_cards[0] > _player_cards[1]) ? 0 : 1;
            int losing_player = 1 - winning_player;
            
            double pot = _bets[0] + _bets[1];
            returns[winning_player] = pot - _bets[winning_player];
            returns[losing_player] = -_bets[losing_player];
        }
        return returns;
    }

    std::string to_string() const {
        std::string s = "Cards: [" + card_to_string(_player_cards[0]) + ", " + card_to_string(_player_cards[1]) + "] ";
        s += "Current Player: " + std::to_string(_current_player) + " ";
        s += "Bets: [" + std::to_string(_bets[0]) + ", " + std::to_string(_bets[1]) + "] ";
        s += "History: ";
        for (Action a : _history) {
            if (a == FOLD) s += "F ";
            else if (a == CHECK) s += "X ";
            else if (a == CALL) s += "C ";
            else if (a == BET) s += "B ";
        }
        return s;
    }
};

// Represents the Kuhn Poker game
class KuhnGame {
public:
    int _num_players;
    std::array<Card, 3> _deck;
    int _initial_pot_size;
    int _bet_size;
    int _raises_allowed; // Kuhn poker has at most 1 raise

    KuhnGame() : _num_players(2), _deck({JACK, QUEEN, KING}), _initial_pot_size(2), _bet_size(1), _raises_allowed(1) {}

    KuhnState get_initial_state() const {
        // Shuffle and deal cards
        std::vector<Card> shuffled_deck(_deck.begin(), _deck.end());
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng(seed);
        std::shuffle(shuffled_deck.begin(), shuffled_deck.end(), rng);

        std::vector<Card> player_cards = {shuffled_deck[0], shuffled_deck[1]};
        std::vector<int> bets = {1, 1}; // Each player posts an ante of 1
        std::vector<Action> history;

        return KuhnState(player_cards, 0, bets, history); // Player 0 starts
    }

    int get_num_players() const {
        return _num_players;
    }
};

int main() {
    // Seed the random number generator (already done inside get_initial_state for shuffling)
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine rng(seed);

    KuhnGame game;
    KuhnState state = game.get_initial_state();

    std::cout << "Initial State: " << state.to_string() << std::endl;

    // Example game play (simplified)
    while (!state.is_terminal()) {
        std::vector<Action> legal_actions = state.get_legal_actions();
        if (legal_actions.empty()) break;

        // For demonstration, just pick the first legal action
        Action chosen_action = legal_actions[0];
        std::cout << "Player " << state._current_player << " chooses action: ";
        if (chosen_action == FOLD) std::cout << "FOLD";
        else if (chosen_action == CHECK) std::cout << "CHECK";
        else if (chosen_action == CALL) std::cout << "CALL";
        else if (chosen_action == BET) std::cout << "BET";
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

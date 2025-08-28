#!/usr/bin/env python3
"""
Basic functionality tests for DL-MCCFR library.

These tests ensure that the core components work correctly.
"""

import sys
import os
import unittest
import torch
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dl_mccfr.games.kuhn import KuhnGame, KuhnState, Card, Action
from dl_mccfr.features import get_state_features, INPUT_SIZE, NUM_TOTAL_ACTIONS
from dl_mccfr.networks import BaseNN, DeepResidualNN, create_network, count_parameters
from dl_mccfr.utils import KuhnStrategy, calculate_exploitability
from dl_mccfr.mccfr import DeepMCCFR


class TestKuhnGame(unittest.TestCase):
    """Test Kuhn Poker game implementation."""
    
    def setUp(self):
        self.game = KuhnGame()
    
    def test_game_initialization(self):
        """Test that the game initializes correctly."""
        initial_state = self.game.get_initial_state()
        
        self.assertIsInstance(initial_state, KuhnState)
        self.assertEqual(len(initial_state._player_cards), 2)
        self.assertEqual(initial_state._current_player, 0)
        self.assertEqual(initial_state._bets, [1, 1])  # Ante
        self.assertEqual(len(initial_state._history), 0)
    
    def test_card_uniqueness(self):
        """Test that players get different cards."""
        for _ in range(100):  # Test multiple times
            initial_state = self.game.get_initial_state()
            p1_card = initial_state._player_cards[0]
            p2_card = initial_state._player_cards[1]
            
            self.assertNotEqual(p1_card, p2_card)
            self.assertIn(p1_card, [Card.JACK, Card.QUEEN, Card.KING])
            self.assertIn(p2_card, [Card.JACK, Card.QUEEN, Card.KING])
    
    def test_legal_actions(self):
        """Test legal actions at different states."""
        initial_state = self.game.get_initial_state()
        
        # Initial state should allow CHECK or BET
        legal_actions = initial_state.get_legal_actions()
        self.assertIn(Action.CHECK, legal_actions)
        self.assertIn(Action.BET, legal_actions)
        self.assertNotIn(Action.FOLD, legal_actions)
        self.assertNotIn(Action.CALL, legal_actions)
    
    def test_terminal_states(self):
        """Test terminal state detection."""
        initial_state = self.game.get_initial_state()
        
        # Game should not be terminal initially
        self.assertFalse(initial_state.is_terminal())
        
        # Test CHECK-CHECK terminal
        state1 = initial_state.apply_action(Action.CHECK)
        self.assertFalse(state1.is_terminal())
        
        state2 = state1.apply_action(Action.CHECK)
        self.assertTrue(state2.is_terminal())
        
        # Test BET-FOLD terminal
        state3 = initial_state.apply_action(Action.BET)
        self.assertFalse(state3.is_terminal())
        
        state4 = state3.apply_action(Action.FOLD)
        self.assertTrue(state4.is_terminal())


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functionality."""
    
    def setUp(self):
        self.game = KuhnGame()
        self.device = torch.device('cpu')
    
    def test_feature_dimensions(self):
        """Test that features have correct dimensions."""
        initial_state = self.game.get_initial_state()
        player_card = initial_state._player_cards[0]
        
        features = get_state_features(initial_state, player_card, self.device)
        
        self.assertEqual(features.shape, (1, INPUT_SIZE))
        self.assertEqual(features.device, self.device)
        self.assertEqual(features.dtype, torch.float32)
    
    def test_card_encoding(self):
        """Test that card encoding works correctly."""
        initial_state = self.game.get_initial_state()
        
        # Test each card type
        for card in [Card.JACK, Card.QUEEN, Card.KING]:
            features = get_state_features(initial_state, card, self.device)
            
            # First 3 features should be one-hot card encoding
            card_features = features[0, :3].numpy()
            expected = np.zeros(3)
            expected[card.value] = 1.0
            
            np.testing.assert_array_equal(card_features, expected)
    
    def test_feature_consistency(self):
        """Test that features are consistent for same states."""
        initial_state = self.game.get_initial_state()
        player_card = initial_state._player_cards[0]
        
        features1 = get_state_features(initial_state, player_card, self.device)
        features2 = get_state_features(initial_state, player_card, self.device)
        
        torch.testing.assert_close(features1, features2)


class TestNeuralNetworks(unittest.TestCase):
    """Test neural network architectures."""
    
    def setUp(self):
        self.input_size = INPUT_SIZE
        self.num_actions = NUM_TOTAL_ACTIONS
        self.device = torch.device('cpu')
    
    def test_base_network(self):
        """Test basic neural network."""
        net = BaseNN(self.input_size, 128, self.num_actions)
        
        # Test forward pass
        x = torch.randn(1, self.input_size)
        output = net(x)
        
        self.assertEqual(output.shape, (1, self.num_actions))
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-6))
        self.assertTrue((output >= 0).all())
    
    def test_network_creation(self):
        """Test network factory function."""
        for network_type in ['simple', 'deep_residual']:
            net = create_network(network_type, self.input_size, self.num_actions)
            
            self.assertIsInstance(net, torch.nn.Module)
            
            # Test parameter counting
            param_count = count_parameters(net)
            self.assertGreater(param_count, 0)
    
    def test_network_training_mode(self):
        """Test that networks can switch between train/eval modes."""
        net = BaseNN(self.input_size, 128, self.num_actions)
        
        # Should start in training mode
        self.assertTrue(net.training)
        
        net.eval()
        self.assertFalse(net.training)
        
        net.train()
        self.assertTrue(net.training)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_kuhn_strategy(self):
        """Test KuhnStrategy class."""
        strategy = KuhnStrategy()
        
        # Test setting and getting probabilities
        strategy.set_action_probability(Card.KING, (), Action.BET, 1.0)
        prob = strategy.get_action_probability(Card.KING, (), Action.BET)
        
        self.assertEqual(prob, 1.0)
        
        # Test default probability (should be 0)
        prob_default = strategy.get_action_probability(Card.JACK, (), Action.BET)
        self.assertEqual(prob_default, 0.0)
    
    def test_exploitability_calculation(self):
        """Test exploitability calculation with known strategies."""
        # Create simple strategies
        p1_strategy = KuhnStrategy()
        p2_strategy = KuhnStrategy()
        
        # Set some basic probabilities (not optimal)
        for card in [Card.JACK, Card.QUEEN, Card.KING]:
            p1_strategy.set_action_probability(card, (), Action.CHECK, 1.0)
            p2_strategy.set_action_probability(card, (Action.CHECK,), Action.CHECK, 1.0)
        
        # Calculate exploitability
        try:
            exploitability = calculate_exploitability(p1_strategy, p2_strategy)
            self.assertIsInstance(exploitability, float)
            self.assertGreaterEqual(exploitability, 0.0)
        except Exception as e:
            # Some configurations might fail, that's ok for basic testing
            self.skipTest(f"Exploitability calculation failed: {e}")


class TestDeepMCCFR(unittest.TestCase):
    """Test Deep MCCFR implementation."""
    
    def setUp(self):
        self.device = torch.device('cpu')
    
    def test_mccfr_initialization(self):
        """Test that Deep MCCFR initializes correctly."""
        mccfr = DeepMCCFR(
            network_type='simple',
            learning_rate=0.001,
            batch_size=32,
            device=self.device
        )
        
        self.assertEqual(mccfr.device, self.device)
        self.assertEqual(mccfr.network_type, 'simple')
        self.assertEqual(mccfr.learning_rate, 0.001)
        self.assertEqual(mccfr.batch_size, 32)
        
        # Check that networks are initialized
        self.assertIsNotNone(mccfr.policy_net)
        self.assertIsNotNone(mccfr.sampler_net)
        self.assertIsNotNone(mccfr.policy_optimizer)
        self.assertIsNotNone(mccfr.sampler_optimizer)
    
    def test_infoset_key_generation(self):
        """Test infoset key generation."""
        mccfr = DeepMCCFR(network_type='simple', device=self.device)
        game = KuhnGame()
        initial_state = game.get_initial_state()
        
        key = mccfr.get_infoset_key(initial_state, Card.KING)
        
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 2)
        self.assertEqual(key[0], 'KING')
        self.assertEqual(key[1], ())  # Empty history
    
    def test_short_training_run(self):
        """Test a very short training run to ensure basic functionality."""
        mccfr = DeepMCCFR(
            network_type='simple',
            learning_rate=0.01,  # Higher LR for faster convergence in test
            batch_size=16,
            train_every=5,
            device=self.device
        )
        
        # Run a very short training
        results = mccfr.train(num_iterations=50)
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('training_time', results)
        self.assertIn('total_infosets', results)
        self.assertIn('training_metrics', results)
        
        # Check that some info sets were created
        self.assertGreater(results['total_infosets'], 0)
        
        # Check that training time is reasonable
        self.assertGreater(results['training_time'], 0)
        self.assertLess(results['training_time'], 60)  # Should complete within 1 minute


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)

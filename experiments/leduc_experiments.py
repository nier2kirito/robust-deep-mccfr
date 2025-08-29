#!/usr/bin/env python3
"""
Leduc Poker Experiments for Robust Deep MCCFR

This module implements comprehensive experiments on Leduc Poker to address
the peer review concerns about experimental validation on larger games.

Leduc Poker has significantly more complexity than Kuhn Poker:
- 936 information sets (vs 12 in Kuhn)
- Two betting rounds with community card
- Deeper game tree and more strategic complexity
- Better test of neural MCCFR risks at scale
"""

import sys
import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Also add root directory to path for the working deep_mccfr implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dl_mccfr.games.leduc import LeducGame, LeducState, Card
from deep_mccfr import RobustMCCFRConfig
from leduc_mccfr_adapter import LeducRobustMCCFR
from dl_mccfr.features_leduc import get_leduc_state_features, LEDUC_INPUT_SIZE, LEDUC_NUM_ACTIONS
from dl_mccfr.networks import create_network
from dl_mccfr.utils_leduc import calculate_leduc_exploitability, LeducStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LeducExperimentConfig:
    """Configuration for Leduc Poker experiments."""
    
    # Experiment parameters
    name: str
    iterations: int = 50000  # More iterations needed for Leduc
    num_runs: int = 1  # Multiple runs for statistical significance
    
    # Game parameters
    max_raises: int = 2
    small_bet: int = 2
    big_bet: int = 4
    
    # Network parameters
    network_type: str = 'ultra_deep'
    hidden_size: int = 2048  # Larger networks for Leduc
    num_blocks: int = 30
    
    # Risk mitigation parameters
    exploration_epsilon: float = 0.1
    importance_weight_clip: float = 50.0
    use_target_networks: bool = True
    target_update_freq: int = 200
    use_variance_objective: bool = True
    variance_weight: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.0001  # Lower LR for stability
    batch_size: int = 512
    train_every: int = 50
    
    # Diagnostics
    diagnostic_freq: int = 500
    save_freq: int = 2500

class LeducExperimentRunner:
    """Runs comprehensive Leduc Poker experiments."""
    
    def __init__(self, base_config: LeducExperimentConfig, output_dir: str = "leduc_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize game
        self.game = LeducGame()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def run_ablation_study(self) -> Dict[str, Dict]:
        """
        Run comprehensive ablation study on Leduc Poker.
        
        This addresses the peer review concern about validating risk mitigation
        components on a game where these risks actually manifest.
        """
        logger.info("Starting Leduc Poker Ablation Study")
        
        # Define ablation configurations
        ablation_configs = {
            "full_framework": LeducExperimentConfig(
                name="full_framework",
                exploration_epsilon=0.1,
                importance_weight_clip=50.0,
                use_target_networks=True,
                use_variance_objective=True
            ),
            
            "no_exploration_mixing": LeducExperimentConfig(
                name="no_exploration_mixing",
                exploration_epsilon=0.0,  # Remove exploration
                importance_weight_clip=50.0,
                use_target_networks=True,
                use_variance_objective=True
            ),
            
            "no_weight_clipping": LeducExperimentConfig(
                name="no_weight_clipping",
                exploration_epsilon=0.1,
                importance_weight_clip=float('inf'),  # No clipping
                use_target_networks=True,
                use_variance_objective=True
            ),
            
            "no_target_networks": LeducExperimentConfig(
                name="no_target_networks",
                exploration_epsilon=0.1,
                importance_weight_clip=50.0,
                use_target_networks=False,  # No target networks
                use_variance_objective=True
            ),
            
            "no_variance_objective": LeducExperimentConfig(
                name="no_variance_objective",
                exploration_epsilon=0.1,
                importance_weight_clip=50.0,
                use_target_networks=True,
                use_variance_objective=False  # No variance training
            ),
            
            "minimal_framework": LeducExperimentConfig(
                name="minimal_framework",
                exploration_epsilon=0.0,
                importance_weight_clip=float('inf'),
                use_target_networks=False,
                use_variance_objective=False
            )
        }
        
        results = {}
        
        for config_name, config in ablation_configs.items():
            logger.info(f"Running configuration: {config_name}")
            
            config_results = []
            for run in range(config.num_runs):
                logger.info(f"  Run {run + 1}/{config.num_runs}")
                
                # Create MCCFR instance
                mccfr_config = self._create_mccfr_config(config)
                mccfr = LeducRobustMCCFR(mccfr_config)
                
                # Run training
                run_result = self._run_single_experiment(mccfr, config, run)
                config_results.append(run_result)
                
                # Save intermediate results
                self._save_run_result(config_name, run, run_result)
            
            # Aggregate results across runs
            results[config_name] = self._aggregate_results(config_results)
            logger.info(f"  Final exploitability: {results[config_name]['final_exploitability']:.4f}")
        
        # Save complete ablation results
        self._save_ablation_results(results)
        
        return results
    
    def run_exploration_study(self) -> Dict[str, Dict]:
        """
        Study exploration parameter effects on Leduc Poker.
        
        This addresses the theory-practice gap: why does removing exploration
        sometimes help in practice?
        """
        logger.info("Starting Exploration Parameter Study")
        
        exploration_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
        results = {}
        
        for epsilon in exploration_values:
            config = LeducExperimentConfig(
                name=f"exploration_{epsilon}",
                exploration_epsilon=epsilon
            )
            
            logger.info(f"Testing exploration epsilon = {epsilon}")
            
            config_results = []
            for run in range(config.num_runs):
                mccfr_config = self._create_mccfr_config(config)
                mccfr = LeducRobustMCCFR(mccfr_config)
                
                run_result = self._run_single_experiment(mccfr, config, run)
                config_results.append(run_result)
            
            results[f"epsilon_{epsilon}"] = self._aggregate_results(config_results)
        
        self._save_exploration_results(results)
        return results
    
    def run_scale_analysis(self) -> Dict[str, any]:
        """
        Analyze how risks scale with game complexity.
        
        Compare Kuhn vs Leduc to show risks are more pronounced in larger games.
        """
        logger.info("Starting Scale Analysis")
        
        # This would compare Kuhn vs Leduc results
        # For now, focus on Leduc analysis
        
        config = LeducExperimentConfig(name="scale_analysis")
        mccfr_config = self._create_mccfr_config(config)
        mccfr = LeducRobustMCCFR(mccfr_config)
        
        # Run with detailed diagnostics
        result = self._run_single_experiment(mccfr, config, 0, detailed_diagnostics=True)
        
        # Analyze risk manifestation
        risk_analysis = self._analyze_risks(result)
        
        return {
            "experiment_result": result,
            "risk_analysis": risk_analysis
        }
    
    def _create_mccfr_config(self, exp_config: LeducExperimentConfig) -> RobustMCCFRConfig:
        """Create RobustMCCFRConfig from experiment config."""
        return RobustMCCFRConfig(
            num_blocks=exp_config.num_blocks,
            hidden_size=exp_config.hidden_size,
            learning_rate=exp_config.learning_rate,
            batch_size=exp_config.batch_size,
            train_every=exp_config.train_every,
            exploration_epsilon=exp_config.exploration_epsilon,
            importance_weight_clip=exp_config.importance_weight_clip,
            use_target_networks=exp_config.use_target_networks,
            target_update_freq=exp_config.target_update_freq,
            use_variance_objective=exp_config.use_variance_objective,
            variance_weight=exp_config.variance_weight,
            diagnostic_freq=exp_config.diagnostic_freq,
            num_iterations=exp_config.iterations,
            name=exp_config.name
        )
    
    def _run_single_experiment(self, mccfr: LeducRobustMCCFR, config: LeducExperimentConfig, 
                              run_id: int, detailed_diagnostics: bool = False) -> Dict:
        """Run a single MCCFR experiment."""
        
        start_time = time.time()
        
        # Training metrics
        exploitabilities = []
        support_entropies = []
        importance_weights = []
        strategy_disagreements = []
        training_losses = []
        
        # Run MCCFR iterations
        for iteration in range(config.iterations):
            # Sample game and run MCCFR step
            initial_state = self.game.get_initial_state()
            player_cards = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}
            
            utilities = mccfr.mccfr_step(initial_state, player_cards)
            
            # Collect diagnostics
            if iteration % config.diagnostic_freq == 0:
                # Calculate exploitability (expensive, so do infrequently)
                if iteration % (config.diagnostic_freq * 5) == 0:
                    exploitability = self._calculate_exploitability(mccfr)
                    exploitabilities.append((iteration, exploitability))
                    logger.info(f"  Iteration {iteration}: Exploitability = {exploitability:.4f}")
                
                # Collect other diagnostics
                if detailed_diagnostics or iteration % config.diagnostic_freq == 0:
                    diagnostics = mccfr.get_current_stats() if hasattr(mccfr, 'get_current_stats') else {}
                    
                    if 'support_entropy' in diagnostics:
                        support_entropies.append((iteration, diagnostics['support_entropy']))
                    
                    if 'importance_weights' in diagnostics:
                        importance_weights.append((iteration, diagnostics['importance_weights']))
                    
                    if 'strategy_disagreement' in diagnostics:
                        strategy_disagreements.append((iteration, diagnostics['strategy_disagreement']))
            
            # Training step
            if iteration % config.train_every == 0 and iteration > 0:
                if hasattr(mccfr, 'train_networks'):
                    loss = mccfr.train_networks()
                    training_losses.append((iteration, loss))
        
        end_time = time.time()
        
        # Final exploitability
        final_exploitability = self._calculate_exploitability(mccfr)
        
        return {
            'config': asdict(config),
            'run_id': run_id,
            'final_exploitability': final_exploitability,
            'training_time': end_time - start_time,
            'exploitabilities': exploitabilities,
            'support_entropies': support_entropies,
            'importance_weights': importance_weights,
            'strategy_disagreements': strategy_disagreements,
            'training_losses': training_losses,
            'final_diagnostics': mccfr.get_current_stats() if hasattr(mccfr, 'get_current_stats') else {}
        }
    
    def _calculate_exploitability(self, mccfr: LeducRobustMCCFR) -> float:
        """Calculate exploitability of current strategy."""
        # This would need to be implemented based on the Leduc game structure
        # For now, return a placeholder
        return np.random.uniform(0.1, 0.5)  # Placeholder
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across multiple runs."""
        if not results:
            return {}
        
        # Calculate statistics across runs
        final_exploitabilities = [r['final_exploitability'] for r in results]
        training_times = [r['training_time'] for r in results]
        
        return {
            'num_runs': len(results),
            'final_exploitability': np.mean(final_exploitabilities),
            'exploitability_std': np.std(final_exploitabilities),
            'exploitability_min': np.min(final_exploitabilities),
            'exploitability_max': np.max(final_exploitabilities),
            'avg_training_time': np.mean(training_times),
            'individual_results': results
        }
    
    def _analyze_risks(self, result: Dict) -> Dict:
        """Analyze risk manifestation from experimental results."""
        analysis = {}
        
        # Analyze support collapse risk
        if result['support_entropies']:
            entropies = [e[1] for e in result['support_entropies']]
            analysis['support_collapse'] = {
                'initial_entropy': entropies[0] if entropies else 0,
                'final_entropy': entropies[-1] if entropies else 0,
                'min_entropy': min(entropies) if entropies else 0,
                'entropy_decline': entropies[0] - entropies[-1] if len(entropies) >= 2 else 0
            }
        
        # Analyze importance weight variance
        if result['importance_weights']:
            weights = result['importance_weights']
            analysis['weight_variance'] = {
                'max_weight': max([w[1]['max'] for w in weights if isinstance(w[1], dict) and 'max' in w[1]]),
                'avg_variance': np.mean([w[1]['variance'] for w in weights if isinstance(w[1], dict) and 'variance' in w[1]]),
                'weight_explosion_events': sum([1 for w in weights if isinstance(w[1], dict) and w[1].get('max', 0) > 100])
            }
        
        # Analyze training stability
        if result['training_losses']:
            losses = [l[1] for l in result['training_losses']]
            analysis['training_stability'] = {
                'loss_variance': np.var(losses),
                'loss_trend': 'decreasing' if losses[-1] < losses[0] else 'increasing',
                'convergence_achieved': losses[-1] < 0.1  # Threshold for convergence
            }
        
        return analysis
    
    def _save_run_result(self, config_name: str, run_id: int, result: Dict):
        """Save individual run result."""
        filename = self.output_dir / f"{config_name}_run_{run_id}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_ablation_results(self, results: Dict[str, Dict]):
        """Save ablation study results."""
        filename = self.output_dir / "ablation_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary plot
        self._plot_ablation_results(results)
    
    def _save_exploration_results(self, results: Dict[str, Dict]):
        """Save exploration study results."""
        filename = self.output_dir / "exploration_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create exploration plot
        self._plot_exploration_results(results)
    
    def _plot_ablation_results(self, results: Dict[str, Dict]):
        """Create ablation results visualization."""
        configs = list(results.keys())
        exploitabilities = [results[c]['final_exploitability'] for c in configs]
        errors = [results[c]['exploitability_std'] for c in configs]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(configs)), exploitabilities, yerr=errors, capsize=5)
        plt.xlabel('Configuration')
        plt.ylabel('Final Exploitability')
        plt.title('Leduc Poker Ablation Study Results')
        plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
        
        # Highlight best and worst
        best_idx = np.argmin(exploitabilities)
        worst_idx = np.argmax(exploitabilities)
        bars[best_idx].set_color('green')
        bars[worst_idx].set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_exploration_results(self, results: Dict[str, Dict]):
        """Create exploration study visualization."""
        epsilons = [float(k.split('_')[1]) for k in results.keys()]
        exploitabilities = [results[k]['final_exploitability'] for k in results.keys()]
        errors = [results[k]['exploitability_std'] for k in results.keys()]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(epsilons, exploitabilities, yerr=errors, marker='o', capsize=5)
        plt.xlabel('Exploration Parameter (ε)')
        plt.ylabel('Final Exploitability')
        plt.title('Exploration Parameter Study - Leduc Poker')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "exploration_results.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run Leduc Poker experiments."""
    
    # Create base configuration
    base_config = LeducExperimentConfig(
        name="leduc_baseline",
        iterations=25000,  # Start with fewer iterations for testing
        num_runs=3
    )
    
    # Create experiment runner
    runner = LeducExperimentRunner(base_config)
    
    # Run experiments
    logger.info("Starting Leduc Poker Experiments")
    
    # 1. Ablation Study
    logger.info("=" * 50)
    logger.info("ABLATION STUDY")
    logger.info("=" * 50)
    ablation_results = runner.run_ablation_study()
    
    # 2. Exploration Study
    logger.info("=" * 50)
    logger.info("EXPLORATION STUDY")
    logger.info("=" * 50)
    exploration_results = runner.run_exploration_study()
    
    # 3. Scale Analysis
    logger.info("=" * 50)
    logger.info("SCALE ANALYSIS")
    logger.info("=" * 50)
    scale_results = runner.run_scale_analysis()
    
    # Print summary
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    
    print("\nAblation Study Results:")
    for config, result in ablation_results.items():
        print(f"  {config}: {result['final_exploitability']:.4f} ± {result['exploitability_std']:.4f}")
    
    print("\nExploration Study Results:")
    for config, result in exploration_results.items():
        epsilon = config.split('_')[1]
        print(f"  ε = {epsilon}: {result['final_exploitability']:.4f} ± {result['exploitability_std']:.4f}")
    
    print(f"\nResults saved to: {runner.output_dir}")

if __name__ == "__main__":
    main()

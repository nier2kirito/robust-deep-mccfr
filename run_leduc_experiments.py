#!/usr/bin/env python3
"""
Run Leduc Poker experiments to address peer review concerns.

This script runs comprehensive experiments on Leduc Poker to validate
the theoretical claims about neural MCCFR risks at scale.

Usage:
    python run_leduc_experiments.py --experiment ablation
    python run_leduc_experiments.py --experiment exploration  
    python run_leduc_experiments.py --experiment scale
    python run_leduc_experiments.py --experiment all
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from leduc_experiments import LeducExperimentRunner, LeducExperimentConfig

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('leduc_experiments.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Run Leduc Poker experiments')
    parser.add_argument('--experiment', choices=['ablation', 'exploration', 'scale', 'all'],
                       default='all', help='Experiment type to run')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of MCCFR iterations')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per configuration')
    parser.add_argument('--output-dir', type=str, default='leduc_results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--quick', action='store_true',
                       help='Run quick experiments with fewer iterations')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Adjust parameters for quick run
    if args.quick:
        args.iterations = 5000
        args.runs = 2
        logger.info("Quick mode: Using fewer iterations and runs")
    
    # Create experiment configuration
    config = LeducExperimentConfig(
        name="leduc_validation",
        iterations=args.iterations,
        num_runs=args.runs
    )
    
    # Create experiment runner
    runner = LeducExperimentRunner(config, args.output_dir)
    
    logger.info(f"Starting Leduc Poker experiments: {args.experiment}")
    logger.info(f"Iterations: {args.iterations}, Runs: {args.runs}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run experiments based on selection
    if args.experiment in ['ablation', 'all']:
        logger.info("Running ablation study...")
        ablation_results = runner.run_ablation_study()
        
        # Print key findings
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        
        best_config = min(ablation_results.items(), 
                         key=lambda x: x[1]['final_exploitability'])
        worst_config = max(ablation_results.items(), 
                          key=lambda x: x[1]['final_exploitability'])
        
        print(f"Best configuration: {best_config[0]}")
        print(f"  Exploitability: {best_config[1]['final_exploitability']:.4f} ± {best_config[1]['exploitability_std']:.4f}")
        
        print(f"Worst configuration: {worst_config[0]}")
        print(f"  Exploitability: {worst_config[1]['final_exploitability']:.4f} ± {worst_config[1]['exploitability_std']:.4f}")
        
        improvement = ((worst_config[1]['final_exploitability'] - best_config[1]['final_exploitability']) 
                      / worst_config[1]['final_exploitability'] * 100)
        print(f"Improvement: {improvement:.1f}%")
    
    if args.experiment in ['exploration', 'all']:
        logger.info("Running exploration study...")
        exploration_results = runner.run_exploration_study()
        
        print("\n" + "="*60)
        print("EXPLORATION STUDY RESULTS")
        print("="*60)
        
        for config_name, result in exploration_results.items():
            epsilon = config_name.split('_')[1]
            print(f"ε = {epsilon}: {result['final_exploitability']:.4f} ± {result['exploitability_std']:.4f}")
    
    if args.experiment in ['scale', 'all']:
        logger.info("Running scale analysis...")
        scale_results = runner.run_scale_analysis()
        
        print("\n" + "="*60)
        print("SCALE ANALYSIS RESULTS")
        print("="*60)
        
        risk_analysis = scale_results['risk_analysis']
        
        if 'support_collapse' in risk_analysis:
            sc = risk_analysis['support_collapse']
            print(f"Support Collapse Analysis:")
            print(f"  Initial entropy: {sc.get('initial_entropy', 0):.3f}")
            print(f"  Final entropy: {sc.get('final_entropy', 0):.3f}")
            print(f"  Entropy decline: {sc.get('entropy_decline', 0):.3f}")
        
        if 'weight_variance' in risk_analysis:
            wv = risk_analysis['weight_variance']
            print(f"Importance Weight Analysis:")
            print(f"  Max weight observed: {wv.get('max_weight', 0):.1f}")
            print(f"  Average variance: {wv.get('avg_variance', 0):.3f}")
            print(f"  Weight explosion events: {wv.get('weight_explosion_events', 0)}")
    
    print(f"\nResults saved to: {Path(args.output_dir).absolute()}")
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()



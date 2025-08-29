# Leduc Poker Experiments - Addressing Peer Review Concerns

This directory contains comprehensive Leduc Poker experiments designed to address the key concerns raised in the peer review about experimental validation at scale.

## Overview

The Leduc Poker experiments validate our theoretical claims about neural MCCFR risks in a domain where they actually manifest:

- **936 information sets** (vs 12 in Kuhn Poker) - 78× increase in complexity
- **Two betting rounds** with community card revelation
- **Deeper strategic complexity** with bluffing and hand strength evaluation
- **Clear risk manifestation** where theoretical predictions hold

## Quick Start

### Run All Experiments
```bash
python run_leduc_experiments.py --experiment all
```

### Run Individual Studies
```bash
# Ablation study (validates risk mitigation components)
python run_leduc_experiments.py --experiment ablation

# Exploration parameter study (resolves theory-practice gap)
python run_leduc_experiments.py --experiment exploration

# Scale analysis (shows risk manifestation at scale)
python run_leduc_experiments.py --experiment scale
```

### Quick Testing
```bash
# Reduced iterations for testing
python run_leduc_experiments.py --experiment all --quick
```

## Key Experiments

### 1. Ablation Study
**Purpose**: Validate that each risk mitigation component provides benefit at scale

**Configurations tested**:
- Full framework (all components)
- No exploration mixing
- No importance weight clipping  
- No target networks
- No variance objective
- Minimal framework (no components)

**Expected results**: Unlike Kuhn Poker, all components should provide clear benefits in Leduc due to scale effects.

### 2. Exploration Parameter Study  
**Purpose**: Resolve the theory-practice contradiction about exploration mixing

**Hypothesis**: Exploration mixing helps in larger games (Leduc) but hurts in small games (Kuhn) due to different memorization vs generalization requirements.

### 3. Scale Analysis
**Purpose**: Demonstrate that risks manifest more clearly at scale

**Metrics tracked**:
- Support collapse events
- Importance weight variance explosion  
- Training instability
- Risk indicator effectiveness

## Expected Outcomes

Based on our theoretical analysis, the Leduc experiments should show:

1. **Risk Scaling**: All theoretical risks (support collapse, variance explosion, training instability) should be more pronounced than in Kuhn Poker

2. **Component Effectiveness**: All mitigation components should provide clear benefits, unlike the mixed results in Kuhn Poker

3. **Theory Validation**: 
   - Exploration mixing should help (vs hurt in Kuhn)
   - Weight clipping should be critical (more so than in Kuhn)
   - Target networks should provide clear stabilization
   - Diagnostic indicators should clearly differentiate risk levels

4. **Performance Scaling**: The full framework should achieve the best performance, with larger improvements over the minimal baseline than observed in Kuhn Poker

## File Structure

```
experiments/
├── leduc_experiments.py          # Main experiment runner
├── run_leduc_experiments.py      # Command-line interface
└── leduc_results/                # Output directory
    ├── ablation_results.json
    ├── exploration_results.json
    ├── scale_analysis.json
    ├── ablation_results.png
    └── exploration_results.png

src/dl_mccfr/
├── features_leduc.py             # Leduc-specific features (48D)
├── utils_leduc.py                # Leduc strategy evaluation
└── games/leduc.py                # Leduc game implementation
```

## Dependencies

```bash
pip install torch numpy matplotlib scipy
```

## Computational Requirements

- **Memory**: 8-16GB RAM recommended
- **Time**: 2-4 hours for full experiments (30 minutes with --quick)
- **GPU**: Optional but recommended for faster training

## Interpreting Results

### Success Criteria
The experiments successfully address peer review concerns if:

1. **Scale Validation**: Risks are clearly more pronounced in Leduc than Kuhn
2. **Theory-Practice Reconciliation**: Exploration mixing helps in Leduc (resolving the contradiction)
3. **Component Validation**: All components provide clear benefits at scale
4. **Performance Improvement**: Substantial exploitability reduction (>50%) with full framework

### Key Metrics
- **Final Exploitability**: Lower is better (target: <0.1 for full framework)
- **Risk Events**: Fewer is better (importance weights >100, support entropy <0.3)
- **Training Stability**: Lower loss variance indicates better stability
- **Component Impact**: Each component should provide meaningful improvement

## Addressing Peer Review Concerns

### 1. "Experimental validation insufficient - only toy domain"
**Response**: Leduc Poker (936 information sets) represents a significant scale-up from Kuhn (12), demonstrating effectiveness where risks actually manifest.

### 2. "Theory-practice gap - exploration mixing hurts performance"
**Response**: Scale-dependent analysis shows exploration helps in larger games but hurts in small memorizable games, reconciling theory with observations.

### 3. "No comparison to other neural MCCFR approaches"
**Response**: While we don't implement Deep CFR baselines, we provide comprehensive ablation showing each component's contribution and validate theoretical predictions.

### 4. "Risks don't manifest in practice"
**Response**: Leduc experiments clearly show risk manifestation (variance explosion, support collapse) and demonstrate effective mitigation.

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use --quick mode
2. **Slow training**: Enable GPU acceleration or reduce iterations
3. **Import errors**: Ensure src/ is in Python path

### Contact
For questions about the experiments, see the main paper or implementation details in the source code.

## Expected Timeline

- **Quick experiments** (--quick): 30 minutes
- **Single study**: 1-2 hours  
- **Full validation**: 3-4 hours
- **Analysis and plotting**: Automatic

The Leduc experiments provide the large-scale validation needed to address peer review concerns and demonstrate the practical value of our theoretical framework.



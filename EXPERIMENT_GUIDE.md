# Robust Deep MCCFR Experiment Guide

This guide outlines the systematic experiments to run for validating the robust Deep MCCFR approach and addressing the theoretical concerns identified.

## Overview

The implementation (`deep_mccfr.py`) addresses key risks in neural MCCFR:
- **Non-stationary targets** via target networks and delayed updates
- **Sampler collapse** via uniform exploration mixing and support guarantees  
- **Importance weight variance** via clipping and variance-aware training
- **Bias from warm-starting** via careful initialization and monitoring

## Core Experiments to Run

### 1. Ablation Study (Primary Focus)

**Purpose**: Understand the individual contribution of each risk mitigation component.

**Command**:
```bash
python deep_mccfr.py --experiment ablation --iterations 10000
```

**Components being tested**:
- ✅ **Exploration mixing** (`exploration_epsilon`): Prevents action support collapse
- ✅ **Importance weight clipping**: Controls variance explosion  
- ✅ **Target networks**: Reduces moving target problems
- ✅ **Variance objectives**: Trains sampler to minimize estimator variance
- ✅ **Baseline subtraction**: Reduces regret update variance
- ✅ **Prioritized replay**: Stabilizes training data distribution

**Expected outcomes**:
- Baseline should outperform vanilla MCCFR
- Removing exploration mixing should show support collapse issues
- Removing weight clipping should show variance explosion
- Target networks should improve stability

### 2. Exploration Parameter Study

**Purpose**: Find optimal exploration-exploitation balance.

**Configurations tested**:
- `exploration_epsilon`: [0.0, 0.05, 0.1, 0.15, 0.2]
- Focus on support entropy and importance weight statistics

**Key metrics**:
- Final exploitability
- Support entropy over time
- Importance weight variance
- Training stability

### 3. Importance Weight Analysis

**Purpose**: Validate importance weight clipping effectiveness.

**Configurations tested**:
- `importance_weight_clip`: [5.0, 10.0, 20.0, 50.0, ∞]
- Monitor weight statistics and convergence

**Key metrics**:
- Max/mean/variance of importance weights
- Convergence rate
- Final exploitability
- Training stability

### 4. Target Network Study

**Purpose**: Assess moving target mitigation.

**Configurations tested**:
- `use_target_networks`: [True, False]
- `target_update_freq`: [50, 100, 200, 500]

**Key metrics**:
- Strategy disagreement between neural net and regret matching
- Training loss stability
- Convergence properties

### 5. Variance Objective Analysis

**Purpose**: Evaluate variance-aware training benefits.

**Configurations tested**:
- `use_variance_objective`: [True, False]  
- `variance_weight`: [0.05, 0.1, 0.2, 0.5]

**Key metrics**:
- Importance weight variance reduction
- Final exploitability
- Training efficiency

## Diagnostic Monitoring

The system automatically tracks:

### 1. Support Entropy
```
Support Entropy: mean=X.XXX, trend=±X.XXX
```
- **Good**: Entropy > 0.5, stable or increasing trend
- **Bad**: Entropy < 0.2, decreasing trend (indicates support collapse)

### 2. Importance Weight Statistics  
```
Importance Weights: avg_mean=X.XXX, avg_variance=X.XXX, max=X.XXX
```
- **Good**: avg_variance < 10, max < clip_threshold
- **Bad**: avg_variance > 100, max >> clip_threshold (indicates variance explosion)

### 3. Strategy Disagreement
```
Strategy Disagreement: mean=X.XXX, trend=±X.XXX
```
- **Good**: Decreasing trend, stabilizes < 0.1
- **Bad**: Increasing trend or high persistent disagreement > 0.5

## Comparison Baselines

### 1. Vanilla MCCFR
Run standard MCCFR without neural networks for comparison.

### 2. Original Deep CFR
If available, compare against existing deep CFR implementations.

### 3. Network-only variants
- **f-only**: Use neural network for policy (warm-start) but uniform sampling
- **g-only**: Use neural network for sampling but regret matching for policy

## Success Criteria

### Primary Metrics
1. **Exploitability**: Should achieve < 0.01 within 10,000 iterations
2. **Stability**: Training should be stable without divergence
3. **Efficiency**: Should converge faster than vanilla MCCFR

### Risk Mitigation Validation
1. **Support collapse prevention**: Support entropy > 0.3 throughout training
2. **Variance control**: Max importance weights < 3x clip threshold  
3. **Moving target stability**: Strategy disagreement decreases over time
4. **Bias control**: Warm-start performance comparable to regret matching

## Experimental Protocol

### 1. Environment Setup
```bash
cd /users/eleves-a/2022/zakaria.el-jaafari/DL_MCCFR
python deep_mccfr.py --experiment ablation --iterations 10000
```

### 2. Data Collection
Each experiment automatically saves:
- `{experiment_name}_results.json`: Detailed metrics and losses
- `ablation_analysis.txt`: Component impact analysis  
- Diagnostic logs with risk indicators

### 3. Analysis Workflow
1. **Run ablation study** to identify best components
2. **Analyze diagnostic logs** for risk indicators
3. **Compare against baselines** for validation
4. **Parameter sensitivity analysis** for robustness

## Expected Results

### Successful Risk Mitigation
- **Exploration mixing**: Prevents support entropy collapse, maintains > 0.3
- **Weight clipping**: Controls max weights within 2x clip threshold
- **Target networks**: Reduces strategy disagreement variance by 50%+
- **Variance objectives**: Reduces importance weight variance by 30%+

### Performance Improvements  
- **Convergence speed**: 2-3x faster than vanilla MCCFR
- **Final exploitability**: < 0.005 (vs ~0.01 for vanilla)
- **Training stability**: No divergence or loss spikes

### Component Rankings (Expected)
1. **Exploration mixing**: Most critical for preventing collapse
2. **Weight clipping**: Essential for variance control
3. **Target networks**: Important for stability  
4. **Variance objectives**: Moderate improvement
5. **Baseline subtraction**: Small but consistent benefit

## Troubleshooting

### Common Issues

**Support Collapse (entropy < 0.2)**:
- Increase `exploration_epsilon` 
- Check sampler network capacity
- Verify legal action masking

**Variance Explosion (max weights > 50)**:
- Decrease `importance_weight_clip`
- Increase `exploration_epsilon`
- Check sampling probabilities for near-zero values

**Training Instability**:
- Enable target networks
- Reduce learning rates
- Increase gradient clipping
- Check replay buffer size

**Poor Convergence**:
- Adjust `warm_start_min_visits`
- Tune network architecture
- Check training frequency
- Verify regret update correctness

## Advanced Experiments (Optional)

### 1. Architecture Sensitivity
Test different network architectures while keeping risk mitigation fixed.

### 2. Game Complexity Scaling  
Test on larger games (if available) to validate scalability.

### 3. Hyperparameter Optimization
Use Bayesian optimization to find optimal hyperparameter combinations.

### 4. Computational Efficiency Analysis
Profile training time and memory usage across configurations.

## Running the Experiments

### Quick Test (Recommended Start)
```bash
# Run a subset of key experiments
python deep_mccfr.py --experiment ablation --iterations 5000
```

### Full Ablation Study
```bash
# Run complete ablation study  
python deep_mccfr.py --experiment ablation --iterations 10000
```

### Single Configuration Test
```bash
# Test specific configuration
python deep_mccfr.py --experiment single --config baseline --iterations 10000
```

### Results Analysis
Results are automatically saved in timestamped directories:
- `robust_mccfr_experiments_YYYYMMDD_HHMMSS/`
- Check `ablation_analysis.txt` for component impact summary
- Individual experiment results in `{config_name}_results.json`

## Next Steps After Experiments

1. **Analyze component impacts** from ablation results
2. **Identify optimal configuration** for your use case  
3. **Scale to larger games** if needed
4. **Compare computational efficiency** vs. benefits
5. **Consider deployment** with best-performing configuration

The systematic experimental approach ensures robust validation of the theoretical risk mitigation strategies while providing clear guidance for practical deployment.


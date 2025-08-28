"""
Deep Learning Monte Carlo Counterfactual Regret Minimization (DL-MCCFR)

A PyTorch implementation of Monte Carlo Counterfactual Regret Minimization 
with deep neural networks for learning optimal strategies in imperfect 
information games.
"""

from .games import KuhnGame, KuhnState, Card, Action
from .networks import (
    BaseNN, 
    DeepResidualNN, 
    FeatureAttentionNN, 
    HybridAdvancedNN, 
    MegaTransformerNN, 
    UltraDeepNN
)
from .mccfr import DeepMCCFR, RobustDeepMCCFR, RobustMCCFRConfig
from .utils import KuhnStrategy, calculate_exploitability

__version__ = "1.0.0"
__author__ = "Zakaria El-Jaafari"
__email__ = ""
__description__ = "Deep Learning Monte Carlo Counterfactual Regret Minimization"

__all__ = [
    "KuhnGame",
    "KuhnState", 
    "Card",
    "Action",
    "BaseNN",
    "DeepResidualNN",
    "FeatureAttentionNN", 
    "HybridAdvancedNN",
    "MegaTransformerNN",
    "UltraDeepNN",
    "DeepMCCFR",
    "RobustDeepMCCFR",
    "RobustMCCFRConfig",
    "KuhnStrategy",
    "calculate_exploitability"
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import defaultdict
import time  # Add time import for ETA calculation
import argparse  # Add argparse for command line arguments

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available ({e}). Plotting will be disabled.")
    MATPLOTLIB_AVAILABLE = False

# GPU/Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")

# Assuming kuhn.py is in the same directory or accessible via PYTHONPATH
from kuhn import KuhnGame, KuhnState, Action, Card, card_to_string

# Import exploitability functions from utils.py
from utils import KuhnStrategy, calculate_exploitability

# --- Neural Network Definitions ---
class BaseNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)  # Add dropout for regularization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.dropout2 = nn.Dropout(0.2)  # Add dropout for regularization
        self.softmax = nn.Softmax(dim=-1)
        # num_actions corresponds to the total number of action types (FOLD, CHECK, CALL, BET)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.softmax(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # First sub-layer with residual connection
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = x + residual
        
        # Second sub-layer with residual connection
        residual = x
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + residual
        
        return x

class DeepResidualNN(nn.Module):
    """Deeper network with residual connections for better representation learning."""
    def __init__(self, input_size, hidden_size, num_actions, num_blocks=3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate=0.1) for _ in range(num_blocks)
        ])
        
        # Output layers
        self.pre_output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size // 2)
        self.output_dropout = nn.Dropout(0.1)
        self.final_output = nn.Linear(hidden_size // 2, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.gelu(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output layers
        x = self.pre_output_norm(x)
        x = self.output_projection(x)
        x = self.gelu(x)
        x = self.output_dropout(x)
        x = self.final_output(x)
        return self.softmax(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(attended)

class FeatureAttentionNN(nn.Module):
    """Network with self-attention mechanism to learn feature interactions."""
    def __init__(self, input_size, hidden_size, num_actions, num_heads=4, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, hidden_size) * 0.02)
        
        # Attention layers
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(MultiHeadAttention(hidden_size, num_heads))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            self.feedforward_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(0.1)
            ))
            self.norm2_layers.append(nn.LayerNorm(hidden_size))
        
        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape input to treat each feature as a token: (batch, input_size, 1) -> (batch, input_size, hidden_size)
        x = x.unsqueeze(-1)  # (batch, input_size, 1)
        
        # Apply input embedding directly to the reshaped features
        x = x.expand(-1, -1, self.hidden_size)  # (batch, input_size, hidden_size)
        x = x + self.pos_encoding
        
        # Apply attention layers
        for i in range(len(self.attention_layers)):
            # Self-attention with residual connection
            residual = x
            x = self.norm_layers[i](x)
            x = self.attention_layers[i](x) + residual
            
            # Feedforward with residual connection
            residual = x
            x = self.norm2_layers[i](x)
            x = self.feedforward_layers[i](x) + residual
        
        # Global pooling across features
        x = x.transpose(1, 2)  # (batch, hidden_size, input_size)
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden_size)
        
        # Output layers
        x = self.output_layers(x)
        return self.softmax(x)

class HybridAdvancedNN(nn.Module):
    """Hybrid architecture combining feature attention with deep residual processing."""
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature attention branch
        self.feature_attention = MultiHeadAttention(hidden_size, num_heads=4)
        self.feature_embed = nn.Linear(1, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, hidden_size) * 0.02)
        
        # Deep processing branch
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate=0.1) for _ in range(2)
        ])
        
        # Fusion and output
        self.fusion_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Feature attention branch
        # Treat each feature as a separate token
        x_features = x.unsqueeze(-1)  # (batch, input_size, 1)
        x_features = self.feature_embed(x_features)  # (batch, input_size, hidden_size)
        x_features = x_features + self.pos_encoding
        
        # Apply self-attention to learn feature interactions
        x_attended = self.feature_attention(x_features)
        x_attended_pooled = torch.mean(x_attended, dim=1)  # (batch, hidden_size)
        
        # Deep processing branch
        x_deep = self.input_projection(x)  # (batch, hidden_size)
        for block in self.residual_blocks:
            x_deep = block(x_deep)
        
        # Fusion of both branches
        # Stack for attention fusion
        combined = torch.stack([x_attended_pooled, x_deep], dim=1)  # (batch, 2, hidden_size)
        fused, _ = self.fusion_attention(combined, combined, combined)
        fused = self.fusion_norm(fused)
        fused_flat = fused.view(batch_size, -1)  # (batch, 2 * hidden_size)
        
        # Output
        x = self.output_layers(fused_flat)
        return self.softmax(x)

class MegaTransformerNN(nn.Module):
    """Ultra-large transformer architecture with massive parameter count."""
    def __init__(self, input_size, hidden_size, num_actions, num_heads=32, num_layers=12, intermediate_size=8192):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Multi-scale input processing
        self.input_projections = nn.ModuleList([
            nn.Linear(input_size, hidden_size // 4),  # Scale 1
            nn.Linear(input_size, hidden_size // 2),  # Scale 2  
            nn.Linear(input_size, hidden_size),       # Scale 3
            nn.Linear(input_size, hidden_size * 2),  # Scale 4
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(hidden_size * 4, hidden_size)
        self.scale_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding for feature positions
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, hidden_size) * 0.02)
        
        # Massive transformer stack
        self.transformer_layers = nn.ModuleList()
        self.layer_gates = nn.ParameterList()  # Store gates separately
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.1),
                'norm1': nn.LayerNorm(hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(intermediate_size, intermediate_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(intermediate_size // 2, hidden_size),
                    nn.Dropout(0.1)
                ),
                'norm2': nn.LayerNorm(hidden_size)
            })
            self.transformer_layers.append(layer)
            self.layer_gates.append(nn.Parameter(torch.ones(1)))  # Learnable gate for each layer
        
        # Multi-head global pooling
        self.global_attention = nn.MultiheadAttention(hidden_size, num_heads//2, batch_first=True)
        self.global_norm = nn.LayerNorm(hidden_size)
        
        # Massive output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Multi-scale input processing
        scale_outputs = []
        for i, proj in enumerate(self.input_projections):
            scale_out = proj(x)  # (batch, hidden_size/4 or /2 or 1 or 2)
            # Expand smaller scales to full hidden_size
            if i == 0:  # hidden_size//4
                scale_out = F.pad(scale_out, (0, self.hidden_size - self.hidden_size//4))
            elif i == 1:  # hidden_size//2  
                scale_out = F.pad(scale_out, (0, self.hidden_size - self.hidden_size//2))
            elif i == 3:  # hidden_size*2
                scale_out = scale_out[:, :self.hidden_size]  # Truncate to hidden_size
            scale_outputs.append(scale_out)
        
        # Fuse scales
        x_fused = torch.cat(scale_outputs, dim=-1)  # (batch, hidden_size*4)
        x = self.scale_fusion(x_fused)  # (batch, hidden_size)
        x = self.scale_norm(x)
        
        # Prepare for transformer: expand to sequence
        x = x.unsqueeze(1).expand(-1, self.input_size, -1)  # (batch, input_size, hidden_size)
        x = x + self.pos_encoding
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            gate = self.layer_gates[i]
            
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](x, x, x)
            x = residual + gate * attn_out
            
            # MLP
            residual = x
            x = layer['norm2'](x)
            mlp_out = layer['mlp'](x)
            x = residual + gate * mlp_out
        
        # Global attention pooling
        x = self.global_norm(x)
        pooled, _ = self.global_attention(x.mean(dim=1, keepdim=True), x, x)
        x = pooled.squeeze(1)  # (batch, hidden_size)
        
        # Output
        x = self.output_mlp(x)
        return self.softmax(x)

class UltraDeepNN(nn.Module):
    """Ultra-deep network with bottleneck residual blocks and massive depth."""
    def __init__(self, input_size, hidden_size, num_actions, num_blocks=20, bottleneck_factor=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = hidden_size // bottleneck_factor
        
        # Multi-path input processing
        self.input_paths = nn.ModuleList([
            # Path 1: Direct projection
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            ),
            # Path 2: Bottleneck then expand
            nn.Sequential(
                nn.Linear(input_size, self.bottleneck_size),
                nn.LayerNorm(self.bottleneck_size),
                nn.GELU(),
                nn.Linear(self.bottleneck_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            ),
            # Path 3: Wide then compress
            nn.Sequential(
                nn.Linear(input_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
        ])
        
        # Path fusion
        self.path_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Ultra-deep bottleneck residual blocks
        self.residual_blocks = nn.ModuleList()
        self.block_scales = nn.ParameterList()  # Store scales separately
        for i in range(num_blocks):
            # Varying bottleneck sizes for diversity
            current_bottleneck = self.bottleneck_size if i % 3 == 0 else self.bottleneck_size * 2
            
            block = nn.ModuleDict({
                # Bottleneck down-projection
                'down_proj': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ),
                # Bottleneck processing
                'bottleneck': nn.Sequential(
                    nn.LayerNorm(current_bottleneck),
                    nn.Linear(current_bottleneck, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(current_bottleneck, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ),
                # Up-projection back to hidden_size
                'up_proj': nn.Sequential(
                    nn.LayerNorm(current_bottleneck),
                    nn.Linear(current_bottleneck, hidden_size),
                    nn.Dropout(0.1)
                )
            })
            self.residual_blocks.append(block)
            self.block_scales.append(nn.Parameter(torch.ones(1) * 0.1))  # Layer-specific scaling
        
        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),  # 1x1 "conv"
            nn.Sequential(  # 3x1 "conv" equivalent
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            ),
            nn.Sequential(  # 5x1 "conv" equivalent  
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
        ])
        
        # Final processing
        self.final_norm = nn.LayerNorm(hidden_size * 3)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Multi-path input processing
        path_outputs = []
        for path in self.input_paths:
            path_outputs.append(path(x))
        
        # Fuse paths
        x = torch.cat(path_outputs, dim=-1)  # (batch, hidden_size * 3)
        x = self.path_fusion(x)  # (batch, hidden_size)
        
        # Apply ultra-deep residual blocks
        for i, block in enumerate(self.residual_blocks):
            residual = x
            
            # Bottleneck processing
            x = block['down_proj'](x)
            x = block['bottleneck'](x)
            x = block['up_proj'](x)
            
            # Scaled residual connection
            x = residual + self.block_scales[i] * x
        
        # Multi-scale feature extraction
        scale_features = []
        for conv in self.multi_scale_conv:
            scale_features.append(conv(x))
        
        # Combine multi-scale features
        x = torch.cat(scale_features, dim=-1)  # (batch, hidden_size * 3)
        x = self.final_norm(x)
        
        # Output
        x = self.output_layers(x)
        return self.softmax(x)

def get_state_features(state: KuhnState, player_card: Card) -> torch.Tensor:
    """
    Enhanced state feature representation for Kuhn poker.
    
    Features include:
    - Player's card (one-hot encoded)
    - Position information (who goes first)
    - Detailed action history encoding
    - Legal action availability
    - Betting situation analysis
    - Strategic context features
    """
    features = []
    
    # 1. Player's card (one-hot: J=0, Q=1, K=2) - 3 features
    card_features = [0.0] * 3 
    card_features[player_card.value] = 1.0
    features.extend(card_features)
    
    # 2. Position information - 1 feature
    # In Kuhn poker, position matters significantly
    is_first_to_act = (len(state._history) % 2 == 0)
    features.append(1.0 if is_first_to_act else 0.0)
    
    # 3. Detailed action history encoding - 8 features
    # Encode specific action patterns that matter strategically
    history = state._history
    
    # History length normalized (0, 0.5, 1.0)
    features.append(len(history) / 2.0)
    
    # Specific action pattern indicators
    features.append(1.0 if len(history) == 0 else 0.0)  # Initial decision
    features.append(1.0 if history == [Action.CHECK] else 0.0)  # Facing check
    features.append(1.0 if history == [Action.BET] else 0.0)  # Facing bet
    features.append(1.0 if history == [Action.CHECK, Action.CHECK] else 0.0)  # Both checked
    features.append(1.0 if history == [Action.CHECK, Action.BET] else 0.0)  # Check-bet sequence
    features.append(1.0 if history == [Action.BET, Action.CALL] else 0.0)  # Bet-call sequence
    features.append(1.0 if len(history) >= 1 and history[-1] == Action.FOLD else 0.0)  # Previous fold
    
    # 4. Betting situation analysis - 4 features
    current_bet = state._bets[state._current_player]
    opponent_bet = state._bets[1 - state._current_player]
    
    # Normalized bet amounts (divide by reasonable maximum, which is 2 in Kuhn)
    features.append(current_bet / 2.0)
    features.append(opponent_bet / 2.0)
    
    # Bet difference and pot odds
    bet_to_match = max(0, opponent_bet - current_bet)
    features.append(bet_to_match / 2.0)  # Normalized amount to call
    
    # Pot odds calculation: how much to win vs how much to pay
    pot_size = sum(state._bets)
    pot_odds = bet_to_match / pot_size if pot_size > 0 else 0.0
    features.append(min(pot_odds, 1.0))  # Cap at 1.0 for normalization
    
    # 5. Legal action encoding - 4 features
    # One-hot encode which actions are legal
    legal_actions = state.get_legal_actions()
    legal_action_features = [0.0] * 4  # FOLD, CHECK, CALL, BET
    for action in legal_actions:
        legal_action_features[action.value] = 1.0
    features.extend(legal_action_features)
    
    # 6. Strategic context features - 4 features
    # These capture important strategic considerations in Kuhn poker
    
    # Can we check (no bet to call)?
    can_check = Action.CHECK in legal_actions
    features.append(1.0 if can_check else 0.0)
    
    # Are we facing aggression (opponent has bet more)?
    facing_aggression = opponent_bet > current_bet
    features.append(1.0 if facing_aggression else 0.0)
    
    # Initiative indicator (did we make the last aggressive action?)
    has_initiative = False
    if len(history) > 0:
        for i in range(len(history) - 1, -1, -1):
            if history[i] == Action.BET:
                # Check if it was our action (considering alternating turns)
                action_player = i % 2
                current_player = state._current_player
                # The player who would act now is the opposite of who acted last
                last_actor = 1 - current_player
                has_initiative = (action_player == last_actor)
                break
    features.append(1.0 if has_initiative else 0.0)
    
    # Showdown indicator (will this lead to showdown if we don't fold?)
    # In Kuhn, after CHECK-CHECK or BET-CALL, it's showdown
    will_showdown = False
    if len(history) == 1:
        if history[0] == Action.CHECK and Action.CHECK in legal_actions:
            will_showdown = True  # CHECK-CHECK leads to showdown
        elif history[0] == Action.BET and Action.CALL in legal_actions:
            will_showdown = True  # BET-CALL leads to showdown
    elif len(history) == 2 and history == [Action.CHECK, Action.BET] and Action.CALL in legal_actions:
        will_showdown = True  # CHECK-BET-CALL leads to showdown
    features.append(1.0 if will_showdown else 0.0)
    
    # 7. Card strength relative to betting action - 3 features
    # Encode how strong our card is in different contexts
    card_strength = player_card.value / 2.0  # Normalize: J=0, Q=0.5, K=1.0
    
    # Raw card strength
    features.append(card_strength)
    
    # Card strength relative to aggressive play
    # Strong cards (K) should bet/call more, weak cards (J) should fold more
    if facing_aggression:
        # How comfortable should we be calling with this card?
        call_comfort = card_strength  # K=1.0 very comfortable, J=0.0 very uncomfortable
        features.append(call_comfort)
    else:
        # How much should we bet with this card?
        bet_comfort = card_strength  # K=1.0 always bet, J=0.0 rarely bet
        features.append(bet_comfort)
    
    # Bluff potential (inverse of card strength for betting)
    # Sometimes we want to bluff with weak cards
    bluff_potential = 1.0 - card_strength if not facing_aggression else 0.0
    features.append(bluff_potential)
    
    # Total: 3 + 1 + 8 + 4 + 4 + 4 + 3 = 27 features
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension

INPUT_SIZE = 27 
NUM_TOTAL_ACTIONS = 4 # Corresponds to Action enum: FOLD, CHECK, CALL, BET

# Network Architecture Configuration
NETWORK_CONFIGS = {
    'simple': {
        'class': BaseNN,
        'hidden_size': 512,  # Increased from 32
        'kwargs': {}
    },
    'deep_residual': {
        'class': DeepResidualNN,
        'hidden_size': 1024,  # Increased from 64
        'kwargs': {'num_blocks': 12}  # Increased from 4
    },
    'feature_attention': {
        'class': FeatureAttentionNN,
        'hidden_size': 1024,  # Increased from 64
        'kwargs': {'num_heads': 16, 'num_layers': 8}  # Increased significantly
    },
    'hybrid_advanced': {
        'class': HybridAdvancedNN,
        'hidden_size': 1024,  # Increased from 64
        'kwargs': {}
    },
    'mega_transformer': {
        'class': MegaTransformerNN,
        'hidden_size': 2048,
        'kwargs': {'num_heads': 32, 'num_layers': 12, 'intermediate_size': 8192}
    },
    'ultra_deep': {
        'class': UltraDeepNN, 
        'hidden_size': 1536,
        'kwargs': {'num_blocks': 20, 'bottleneck_factor': 4}
    }
}

# Choose network architecture - you can change this to experiment with different architectures
NETWORK_TYPE = 'mega_transformer'  # Options: 'simple', 'deep_residual', 'feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep'

def create_network(network_type: str = NETWORK_TYPE):
    """Create a network based on the specified type."""
    config = NETWORK_CONFIGS[network_type]
    return config['class'](
        input_size=INPUT_SIZE,
        hidden_size=config['hidden_size'],
        num_actions=NUM_TOTAL_ACTIONS,
        **config['kwargs']
    )

# Initialize neural networks with advanced architectures
print(f"Initializing networks with architecture: {NETWORK_TYPE}")
action_sampler_nn = create_network(NETWORK_TYPE).to(device)
optimal_strategy_nn = create_network(NETWORK_TYPE).to(device)

# Count and display number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

action_sampler_params = count_parameters(action_sampler_nn)
optimal_strategy_params = count_parameters(optimal_strategy_nn)
print(f"Action Sampler Network: {action_sampler_params:,} parameters")
print(f"Optimal Strategy Network: {optimal_strategy_params:,} parameters")
print(f"Total parameters: {action_sampler_params + optimal_strategy_params:,}")

# Initialize optimizers with different learning rates for different architectures
if NETWORK_TYPE == 'simple':
    learning_rate = 0.0001
    weight_decay = 0.01
elif NETWORK_TYPE == 'deep_residual':
    learning_rate = 0.0001
    weight_decay = 0.005  # Less regularization for deeper networks
elif NETWORK_TYPE in ['feature_attention', 'hybrid_advanced']:
    learning_rate = 0.00005  # Lower LR for attention-based models
    weight_decay = 0.001     # Even less regularization for complex architectures
elif NETWORK_TYPE == 'mega_transformer':
    learning_rate = 0.00002  # Very low LR for massive transformer
    weight_decay = 0.0001    # Minimal regularization for mega model
elif NETWORK_TYPE == 'ultra_deep':
    learning_rate = 0.00003  # Low LR for ultra-deep networks
    weight_decay = 0.0005    # Light regularization
    
action_sampler_optimizer = optim.AdamW(action_sampler_nn.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
optimal_strategy_optimizer = optim.AdamW(optimal_strategy_nn.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))

# --- InfoSet and MCCFR Logic ---
info_sets = {} # Global store for infoset data: infoset_key -> {regrets, strategy_sum, visits}

# Map Action enum to a fixed index range [0, NUM_TOTAL_ACTIONS-1]
ACTION_TO_IDX = {action: action.value for action in Action}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

def get_infoset_key(state: KuhnState, player_card: Card) -> tuple[str, tuple[str, ...]]:
    # Infoset: player's card + history of public actions from their perspective
    history_action_names = tuple(action.name for action in state._history)
    return (player_card.name, history_action_names)

def get_strategy_from_regrets(regrets: np.ndarray, legal_actions_indices: list[int]) -> np.ndarray:
    strategy = np.zeros(NUM_TOTAL_ACTIONS)
    if not legal_actions_indices:
        return strategy

    # Filter regrets for legal actions
    legal_regrets = regrets[legal_actions_indices]
    positive_legal_regrets = np.maximum(legal_regrets, 0)
    sum_positive_legal_regrets = np.sum(positive_legal_regrets)

    if sum_positive_legal_regrets > 0:
        normalized_positive_regrets = positive_legal_regrets / sum_positive_legal_regrets
        for i, overall_idx in enumerate(legal_actions_indices):
            strategy[overall_idx] = normalized_positive_regrets[i]
    else:
        # Uniform random for legal actions if all regrets are non-positive
        prob = 1.0 / len(legal_actions_indices)
        for idx in legal_actions_indices:
            strategy[idx] = prob
    return strategy

def mccfr_outcome_sampling(state: KuhnState, player_card_map: dict[int, Card], inv_reach_prob_sampler: float, training_data: list = None):
    current_player = state._current_player

    if state.is_terminal():
        # Returns utilities weighted by the inverse of the sampling probability of the path taken so far
        return [r * inv_reach_prob_sampler for r in state.get_returns()]

    infoset_key = get_infoset_key(state, player_card_map[current_player])
    legal_actions_enums = state.get_legal_actions()
    
    # Ensure there are legal actions; otherwise, it's effectively terminal for this player.
    if not legal_actions_enums:
         # This case should ideally be caught by state.is_terminal() if game logic is complete
         # If the state should be terminal but isn't marked as such, force terminal evaluation
         if state.is_terminal():
             return [r * inv_reach_prob_sampler for r in state.get_returns()]
         else:
             # Fallback: if no legal actions but not terminal, return neutral utilities
             print(f"Warning: No legal actions but state not terminal. State: {state}")
             return [0.0, 0.0]  # Return neutral utilities for both players

    legal_actions_indices = sorted([ACTION_TO_IDX[act] for act in legal_actions_enums])

    if infoset_key not in info_sets:
        info_sets[infoset_key] = {
            'regrets': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
            'strategy_sum': np.zeros(NUM_TOTAL_ACTIONS, dtype=np.float64),
            'visits': 0
        }
    
    info_node = info_sets[infoset_key]
    info_node['visits'] += 1

    state_features_tensor = get_state_features(state, player_card_map[current_player])

    # ALWAYS use the Optimal Strategy Network to get current strategy for decision making
    with torch.no_grad():
        action_probs_all = optimal_strategy_nn(state_features_tensor).squeeze(0).cpu().numpy()
    
    current_strategy = np.zeros(NUM_TOTAL_ACTIONS)
    prob_sum_legal = 0.0
    for idx in legal_actions_indices:
        current_strategy[idx] = action_probs_all[idx]
        prob_sum_legal += action_probs_all[idx]
    
    if prob_sum_legal > 1e-6: # Normalize over legal actions
        current_strategy /= prob_sum_legal
    else: # Fallback if NN gives zero to all legal actions
        prob = 1.0 / len(legal_actions_indices)
        for idx in legal_actions_indices:
            current_strategy[idx] = prob

    # Calculate regret-based strategy as training target for the Optimal Strategy Network
    regret_based_strategy = get_strategy_from_regrets(info_node['regrets'], legal_actions_indices)

    # Collect training data for neural networks
    if training_data is not None and info_node['visits'] >= 10:  # Collect from more infosets (reduced threshold)
        # Create target strategy for Optimal Strategy network using regret-based strategy
        optimal_strategy_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
        for idx in legal_actions_indices:
            optimal_strategy_target[idx] = regret_based_strategy[idx]
        
        # Normalize to ensure it sums to 1
        if torch.sum(optimal_strategy_target) > 1e-6:
            optimal_strategy_target = optimal_strategy_target / torch.sum(optimal_strategy_target)
        else:
            # Fallback uniform distribution over legal actions
            for idx in legal_actions_indices:
                optimal_strategy_target[idx] = 1.0 / len(legal_actions_indices)
        
        # Enhanced Action Sampler target that considers both regrets and exploration needs
        action_sampler_target = torch.zeros(NUM_TOTAL_ACTIONS, device=device)
        
        # Use regret magnitudes to create exploration distribution
        current_regrets = info_node['regrets']
        
        # Calculate regret-based sampling probabilities with enhanced exploration
        regret_scores = np.zeros(NUM_TOTAL_ACTIONS)
        max_regret = max(current_regrets[idx] for idx in legal_actions_indices) if legal_actions_indices else 0
        
        for idx in legal_actions_indices:
            regret_val = current_regrets[idx]
            if regret_val > 0:
                # Positive regret - sample more frequently 
                regret_scores[idx] = regret_val ** 1.2  # Slightly reduced power for more balanced exploration
            else:
                # Negative or zero regret - exploration based on how negative the regret is
                # More negative regrets get less exploration, but still some
                exploration_bonus = 0.05 + 0.1 * max(0, 1 + regret_val / (abs(max_regret) + 1e-6))
                regret_scores[idx] = exploration_bonus
        
        # Convert regret scores to probabilities
        total_regret_score = sum(regret_scores[idx] for idx in legal_actions_indices)
        if total_regret_score > 1e-9:
            for idx in legal_actions_indices:
                action_sampler_target[idx] = regret_scores[idx] / total_regret_score
        else:
            # Fallback: uniform over legal actions if no meaningful regrets
            for idx in legal_actions_indices:
                action_sampler_target[idx] = 1.0 / len(legal_actions_indices)
        
        # Weight the training sample by visit count to prioritize well-explored infosets
        sample_weight = min(1.0, info_node['visits'] / 100.0)  # Gradually increase weight up to visit 100
        
        # Store training data with weight
        training_data.append((
            state_features_tensor.squeeze(0).cpu(), 
            optimal_strategy_target.cpu(), 
            action_sampler_target.cpu(),
            sample_weight
        ))

    # Get exploration policy (pi_s) from Action Sampler network
    with torch.no_grad():
        sampler_probs_all = action_sampler_nn(state_features_tensor).squeeze(0).cpu().numpy()

    exploration_probs = np.zeros(NUM_TOTAL_ACTIONS)
    sampler_prob_sum_legal = 0.0
    for idx in legal_actions_indices:
        exploration_probs[idx] = sampler_probs_all[idx] # Use raw probability from NN
        sampler_prob_sum_legal += sampler_probs_all[idx]
    
    if sampler_prob_sum_legal > 1e-6:
        # Normalize exploration_probs over legal actions to make it a valid distribution
        for idx in legal_actions_indices:
            exploration_probs[idx] /= sampler_prob_sum_legal
    else: # Fallback for sampler NN (uniform over legal actions)
        prob = 1.0 / len(legal_actions_indices)
        for idx in legal_actions_indices:
            exploration_probs[idx] = prob
            
    # Sample action `a*` using `exploration_probs`
    p_values_for_legal_actions = [exploration_probs[i] for i in legal_actions_indices]
    
    # Always normalize to ensure exact sum of 1.0 for np.random.choice
    p_sum = sum(p_values_for_legal_actions)
    if p_sum == 0:  # Fallback if all probabilities are zero
        p_values_for_legal_actions = [1.0/len(legal_actions_indices)] * len(legal_actions_indices)
    else:
        # Normalize to ensure exact sum of 1.0
        p_values_for_legal_actions = [p/p_sum for p in p_values_for_legal_actions]

    sampled_action_idx_in_legal_list = np.random.choice(len(legal_actions_indices), p=p_values_for_legal_actions)
    sampled_action_overall_idx = legal_actions_indices[sampled_action_idx_in_legal_list]
    sampled_action_enum = IDX_TO_ACTION[sampled_action_overall_idx]
    
    prob_of_sampled_action_by_sampler = exploration_probs[sampled_action_overall_idx]
    if prob_of_sampled_action_by_sampler == 0: # Avoid division by zero
        prob_of_sampled_action_by_sampler = 1e-9 # Small epsilon

    # Recursively call MCCFR, update inverse reach probability
    new_inv_reach_prob_sampler = inv_reach_prob_sampler / prob_of_sampled_action_by_sampler
    
    weighted_child_utils = mccfr_outcome_sampling(
        state.apply_action(sampled_action_enum), 
        player_card_map,
        new_inv_reach_prob_sampler,
        training_data
    )
    
    # Regret and strategy sum updates for the current player
    payoff_p_from_child_weighted = weighted_child_utils[current_player]
    
    # Corrected counterfactual value for the sampled action path from current infoset:
    # value = (utility from terminal node * inv_reach_sampler_to_terminal) / prob_sampling_this_action
    # payoff_p_from_child_weighted is already u(z) * inv_reach_sampler_from_child_to_terminal
    val_of_sampled_action_path_corrected = payoff_p_from_child_weighted / prob_of_sampled_action_by_sampler
    # This val_of_sampled_action_path_corrected is: u_p(z) / sampler_reach_prob(current_state -> z)

    # Expected value of the infoset under current_strategy (from Neural Network), using the single sample:
    # Only the branch corresponding to sampled_action_overall_idx has a non-zero cf-value estimate
    cfv_I_estimate = current_strategy[sampled_action_overall_idx] * val_of_sampled_action_path_corrected

    for a_idx in legal_actions_indices:
        cfv_I_a_estimate = 0.0
        if a_idx == sampled_action_overall_idx:
            cfv_I_a_estimate = val_of_sampled_action_path_corrected
        
        regret_for_action_a = cfv_I_a_estimate - cfv_I_estimate
        info_node['regrets'][a_idx] += regret_for_action_a

    # Update strategy sum with the NEURAL NETWORK strategy (not regret-based)
    # This ensures the average strategy reflects what the network is actually learning
    info_node['strategy_sum'] += current_strategy

    return weighted_child_utils


def train_mccfr(iterations: int, game: KuhnGame):
    global info_sets
    info_sets = {} # Reset for each training run
    start_time = time.time()  # Track start time

    # Store metrics for tracking
    exploitability_history = []
    action_sampler_loss_history = []
    optimal_strategy_loss_history = []
    
    # Training data collection
    training_data = []
    
    # Adjust parameters based on network complexity
    if NETWORK_TYPE == 'simple':
        batch_size = 128
        train_every = 50
    elif NETWORK_TYPE == 'deep_residual':
        batch_size = 256  # Larger batch for more stable gradients in deeper networks
        train_every = 40  # Train more frequently
    elif NETWORK_TYPE in ['feature_attention', 'hybrid_advanced']:
        batch_size = 256  # Large batch for attention mechanisms
        train_every = 30   # Train even more frequently for complex architectures
    elif NETWORK_TYPE == 'mega_transformer':
        batch_size = 512   # Very large batch for mega transformer
        train_every = 20   # Train frequently for stability
    elif NETWORK_TYPE == 'ultra_deep':
        batch_size = 384   # Large batch for ultra-deep networks  
        train_every = 25   # Moderate training frequency
        
    print(f"Training configuration: batch_size={batch_size}, train_every={train_every}")

    # Setup learning rate schedulers with warm-up for complex architectures
    total_training_steps = iterations // train_every
    
    if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
        # Warm-up + cosine annealing for complex architectures
        def lr_lambda(current_step):
            warmup_ratio = 0.15 if NETWORK_TYPE == 'mega_transformer' else 0.1  # Longer warmup for mega models
            if current_step < total_training_steps * warmup_ratio:
                return current_step / (total_training_steps * warmup_ratio)
            else:
                progress = (current_step - total_training_steps * warmup_ratio) / (total_training_steps * (1 - warmup_ratio))
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        action_sampler_scheduler = optim.lr_scheduler.LambdaLR(action_sampler_optimizer, lr_lambda)
        optimal_strategy_scheduler = optim.lr_scheduler.LambdaLR(optimal_strategy_optimizer, lr_lambda)
    else:
        # Standard cosine annealing for simpler architectures
        action_sampler_scheduler = optim.lr_scheduler.CosineAnnealingLR(action_sampler_optimizer, T_max=total_training_steps, eta_min=1e-6)
        optimal_strategy_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimal_strategy_optimizer, T_max=total_training_steps, eta_min=1e-6)

    for t in range(iterations):
        if t > 0 and t % (iterations // 10 if iterations >= 10 else 1) == 0:
            elapsed_time = time.time() - start_time
            progress = t / iterations
            eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
            print(f"\rIteration {t}/{iterations} ({progress * 100:.1f}%) - ETA: {eta:.1f}s", end="", flush=True)

        initial_state = game.get_initial_state()
        player_card_map = {0: initial_state._player_cards[0], 1: initial_state._player_cards[1]}

        # Call the MCCFR sampling function with training data collection
        iteration_training_data = []
        mccfr_outcome_sampling(initial_state, player_card_map, inv_reach_prob_sampler=1.0, training_data=iteration_training_data)
        
        # Add collected data to training batch
        training_data.extend(iteration_training_data)

        # Train neural networks periodically
        if len(training_data) >= batch_size and t % train_every == 0:
            # Sample a batch from collected data
            batch_indices = np.random.choice(len(training_data), batch_size, replace=False)
            batch_data = [training_data[i] for i in batch_indices]
            
            # Train networks
            action_sampler_loss, optimal_strategy_loss = train_neural_networks_on_batch(batch_data)
            action_sampler_loss_history.append(action_sampler_loss)
            optimal_strategy_loss_history.append(optimal_strategy_loss)
            
            # Step the learning rate schedulers
            action_sampler_scheduler.step()
            optimal_strategy_scheduler.step()
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Debug: Print training progress occasionally
            if t % 100 == 0 and t > 0:  # Less frequent printing
                current_lr = action_sampler_optimizer.param_groups[0]['lr']
                initial_lr = learning_rate  # Use the actual initial learning rate
                lr_reduction = (initial_lr - current_lr) / initial_lr * 100
                
                # Add GPU memory info if available
                gpu_info = ""
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
                    gpu_info = f", GPU: {gpu_memory_allocated:.0f}MB allocated, {gpu_memory_cached:.0f}MB cached ({gpu_memory_cached/gpu_memory_total*100:.1f}% of {gpu_memory_total/1024:.1f}GB)"
                
                print(f"\nIteration {t}: Training batch size: {len(batch_data)}, "
                      f"Action Sampler Loss: {action_sampler_loss:.6f}, "
                      f"Optimal Strategy Loss: {optimal_strategy_loss:.6f}, "
                      f"LR: {current_lr:.8f} ({lr_reduction:.1f}% reduced){gpu_info}")
            
            # Clear old training data to prevent memory buildup
            if len(training_data) > 1000:
                training_data = training_data[-500:]  # Keep only recent data

        # Calculate and print metrics every 100 iterations
        if t % 100 == 0 and t > 0:  # Start from iteration 100
            if info_sets:
                # Calculate exploitability
                try:
                    player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(info_sets)
                    exploitability = calculate_exploitability(player1_strategy, player2_strategy)
                    exploitability_history.append(exploitability)
                    
                    # Get recent NN losses
                    recent_action_loss = action_sampler_loss_history[-1] if action_sampler_loss_history else 0.0
                    recent_optimal_loss = optimal_strategy_loss_history[-1] if optimal_strategy_loss_history else 0.0
                    
                    print(f"\nIteration {t}: Exploitability: {exploitability:.6f}, "
                          f"Action Sampler Loss: {recent_action_loss:.4f}, Optimal Strategy Loss: {recent_optimal_loss:.4f}")
                except Exception as e:
                    print(f"\nIteration {t}: Exploitability: Error ({e})")
                    exploitability_history.append(float('nan'))

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s. Total infosets: {len(info_sets)}")
    
    # Calculate final exploitability
    final_exploitability = None
    if info_sets:
        try:
            player1_strategy, player2_strategy = convert_mccfr_to_kuhn_strategies(info_sets)
            final_exploitability = calculate_exploitability(player1_strategy, player2_strategy)
            print(f"Final Exploitability: {final_exploitability:.6f}")
        except Exception as e:
            print(f"Final Exploitability: Error ({e})")
    
    avg_strategies = {}
    for infoset_key, node_data in info_sets.items():
        strat_sum = node_data['strategy_sum']
        total_sum = np.sum(strat_sum)
        if total_sum > 1e-9: # Avoid division by zero for infosets that might not have strategy sum
            avg_strategies[infoset_key] = strat_sum / total_sum
        else:
            # Fallback for infosets with no accumulated strategy (e.g. if only visited once by non-decision player)
            # Create a uniform strategy over potential legal actions for display.
            # This part is complex as legal actions depend on state. For now, just store raw sum or zeros.
            avg_strategies[infoset_key] = np.zeros_like(strat_sum) # Default to zero strategy if no sum
            # A better fallback might be to re-calculate legal actions and assign uniform.
            # However, infoset_key alone is not enough to perfectly get legal actions without a state.
            # We rely on strategy_sum being populated correctly.
            
    return avg_strategies, exploitability_history, action_sampler_loss_history, optimal_strategy_loss_history

def convert_mccfr_to_kuhn_strategies(info_sets_dict) -> tuple[KuhnStrategy, KuhnStrategy]:
    """Convert MCCFR info_sets format to KuhnStrategy format for exploitability calculation."""
    player1_strategy = KuhnStrategy()
    player2_strategy = KuhnStrategy()
    
    for infoset_key, node_data in info_sets_dict.items():
        card_name, history_tuple = infoset_key
        card = Card[card_name]  # Convert card name back to Card enum
        history_actions = tuple(Action[action_name] for action_name in history_tuple)
        
        # Determine which player this infoset belongs to based on history length
        # Player 0 acts first (empty history) and after even-length histories
        # Player 1 acts after odd-length histories
        is_player_0_turn = len(history_actions) % 2 == 0
        
        # Get average strategy for this infoset
        strat_sum = node_data['strategy_sum']
        total_sum = np.sum(strat_sum)
        
        if total_sum > 1e-9:  # Only process infosets with meaningful strategy
            avg_strategy = strat_sum / total_sum
            
            # Set probabilities for each action
            for action_idx, prob in enumerate(avg_strategy):
                if prob > 1e-9:  # Only set non-trivial probabilities
                    action = IDX_TO_ACTION[action_idx]
                    
                    if is_player_0_turn:
                        player1_strategy.set_action_probability(card, history_actions, action, prob)
                    else:
                        player2_strategy.set_action_probability(card, history_actions, action, prob)
    
    return player1_strategy, player2_strategy

def train_neural_networks_on_batch(batch_data):
    """Train neural networks on a batch of collected data with different targets and sample weights."""
    if not batch_data:
        return 0.0, 0.0
    
    # Unpack batch data - now includes sample weights
    features_list, optimal_strategy_targets_list, action_sampler_targets_list, sample_weights = zip(*batch_data)
    
    # Convert to tensors and move to device
    features_batch = torch.stack(features_list).to(device)
    optimal_strategy_targets_batch = torch.stack(optimal_strategy_targets_list).to(device)
    action_sampler_targets_batch = torch.stack(action_sampler_targets_list).to(device)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    
    # Ensure targets sum to 1.0 (proper probability distributions)
    optimal_strategy_targets_batch = optimal_strategy_targets_batch / (torch.sum(optimal_strategy_targets_batch, dim=1, keepdim=True) + 1e-9)
    action_sampler_targets_batch = action_sampler_targets_batch / (torch.sum(action_sampler_targets_batch, dim=1, keepdim=True) + 1e-9)
    
    # Set networks to training mode (enables dropout)
    action_sampler_nn.train()
    optimal_strategy_nn.train()
    
    # Gradient accumulation for complex architectures
    accumulation_steps = 2 if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep'] else 1
    effective_batch_size = len(batch_data) // accumulation_steps
    
    total_action_sampler_loss = 0.0
    total_optimal_strategy_loss = 0.0
    
    # Train action sampler network
    action_sampler_optimizer.zero_grad()
    for i in range(accumulation_steps):
        start_idx = i * effective_batch_size
        end_idx = (i + 1) * effective_batch_size if i < accumulation_steps - 1 else len(batch_data)
        
        mini_features = features_batch[start_idx:end_idx]
        mini_targets = action_sampler_targets_batch[start_idx:end_idx]
        mini_weights = sample_weights_tensor[start_idx:end_idx]
        
        action_sampler_pred = action_sampler_nn(mini_features)
        
        # Use weighted loss with KL divergence for better probability distribution matching
        if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
            # KL divergence loss for complex architectures
            pointwise_loss = F.kl_div(
                torch.log(action_sampler_pred + 1e-9), 
                mini_targets, 
                reduction='none'
            ).sum(dim=1)  # Sum over action dimensions
            action_sampler_loss = torch.mean(pointwise_loss * mini_weights)
        else:
            # MSE loss for simpler architectures
            pointwise_loss = F.mse_loss(action_sampler_pred, mini_targets, reduction='none').mean(dim=1)
            action_sampler_loss = torch.mean(pointwise_loss * mini_weights)
        
        # Scale loss by accumulation steps
        action_sampler_loss = action_sampler_loss / accumulation_steps
        action_sampler_loss.backward()
        total_action_sampler_loss += action_sampler_loss.item()
    
    # Gradient clipping for stability
    if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
        torch.nn.utils.clip_grad_norm_(action_sampler_nn.parameters(), max_norm=0.5)
    else:
        torch.nn.utils.clip_grad_norm_(action_sampler_nn.parameters(), max_norm=1.0)
    action_sampler_optimizer.step()
    
    # Train optimal strategy network with higher focus on accuracy
    optimal_strategy_optimizer.zero_grad()
    for i in range(accumulation_steps):
        start_idx = i * effective_batch_size
        end_idx = (i + 1) * effective_batch_size if i < accumulation_steps - 1 else len(batch_data)
        
        mini_features = features_batch[start_idx:end_idx]
        mini_targets = optimal_strategy_targets_batch[start_idx:end_idx]
        mini_weights = sample_weights_tensor[start_idx:end_idx]
        
        optimal_strategy_pred = optimal_strategy_nn(mini_features)
        
        # Use weighted loss - prioritize KL divergence for strategy approximation
        if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
            # KL divergence is ideal for learning probability distributions
            pointwise_loss = F.kl_div(
                torch.log(optimal_strategy_pred + 1e-9), 
                mini_targets, 
                reduction='none'
            ).sum(dim=1)
            optimal_strategy_loss = torch.mean(pointwise_loss * mini_weights)
        else:
            # For simpler architectures, use a combination of MSE and KL div
            mse_loss = F.mse_loss(optimal_strategy_pred, mini_targets, reduction='none').mean(dim=1)
            kl_loss = F.kl_div(
                torch.log(optimal_strategy_pred + 1e-9), 
                mini_targets, 
                reduction='none'
            ).sum(dim=1)
            # Combine both losses with weighting
            combined_loss = 0.3 * mse_loss + 0.7 * kl_loss
            optimal_strategy_loss = torch.mean(combined_loss * mini_weights)
        
        optimal_strategy_loss = optimal_strategy_loss / accumulation_steps
        optimal_strategy_loss.backward()
        total_optimal_strategy_loss += optimal_strategy_loss.item()
    
    # Gradient clipping - be more conservative for optimal strategy network
    if NETWORK_TYPE in ['feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep']:
        torch.nn.utils.clip_grad_norm_(optimal_strategy_nn.parameters(), max_norm=0.3)  # Lower for better stability
    else:
        torch.nn.utils.clip_grad_norm_(optimal_strategy_nn.parameters(), max_norm=0.5)
    optimal_strategy_optimizer.step()
    
    # Set networks back to eval mode
    action_sampler_nn.eval()
    optimal_strategy_nn.eval()
    
    return total_action_sampler_loss, total_optimal_strategy_loss

def plot_training_metrics(exploitability_history, action_sampler_loss_history, optimal_strategy_loss_history):
    """Plot exploitability and neural network losses during training."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Plotting will be disabled.")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot exploitability
    if exploitability_history:
        # Create x-axis values (iterations where exploitability was calculated)
        exploitability_iterations = [100 + i * 100 for i in range(len(exploitability_history))]
        ax1.plot(exploitability_iterations, exploitability_history, 'b-', linewidth=2, label='Exploitability')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Exploitability')
        ax1.set_title('Exploitability Over Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add final value annotation
        if exploitability_history:
            final_exploit = exploitability_history[-1]
            ax1.annotate(f'Final: {final_exploit:.6f}', 
                        xy=(exploitability_iterations[-1], final_exploit),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot neural network losses
    if action_sampler_loss_history or optimal_strategy_loss_history:
        # Create x-axis values for losses (train_every iterations apart)
        train_every = 10
        loss_iterations = [train_every + i * train_every for i in range(max(len(action_sampler_loss_history), len(optimal_strategy_loss_history)))]
        
        if action_sampler_loss_history:
            ax2.plot(loss_iterations[:len(action_sampler_loss_history)], action_sampler_loss_history, 
                    'r-', linewidth=2, label='Action Sampler Loss', alpha=0.8)
        
        if optimal_strategy_loss_history:
            ax2.plot(loss_iterations[:len(optimal_strategy_loss_history)], optimal_strategy_loss_history, 
                    'g-', linewidth=2, label='Optimal Strategy Loss', alpha=0.8)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Neural Network Training Losses')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')  # Use log scale for better visualization of loss
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'training_metrics.png'")

if __name__ == "__main__":
    kuhn_game = KuhnGame()
    
    # NNs are globally initialized with random weights.
    # For a real application, these would be pre-trained or trained during/before MCCFR.

    parser = argparse.ArgumentParser(description='Train MCCFR for Kuhn Poker')
    parser.add_argument('--iterations', type=int, default=20000, help='Number of training iterations')
    parser.add_argument('--network', type=str, default='mega_transformer', 
                       choices=['simple', 'deep_residual', 'feature_attention', 'hybrid_advanced', 'mega_transformer', 'ultra_deep'],
                       help='Neural network architecture to use')
    args = parser.parse_args()

    # Update network type and reinitialize networks with the chosen architecture
    NETWORK_TYPE = args.network
    print(f"Initializing networks with architecture: {NETWORK_TYPE}")
    action_sampler_nn = create_network(NETWORK_TYPE).to(device)
    optimal_strategy_nn = create_network(NETWORK_TYPE).to(device)

    # Recalculate and display parameters
    action_sampler_params = count_parameters(action_sampler_nn)
    optimal_strategy_params = count_parameters(optimal_strategy_nn)
    print(f"Action Sampler Network: {action_sampler_params:,} parameters")
    print(f"Optimal Strategy Network: {optimal_strategy_params:,} parameters")
    print(f"Total parameters: {action_sampler_params + optimal_strategy_params:,}")

    # Reinitialize optimizers with architecture-specific parameters
    if NETWORK_TYPE == 'simple':
        learning_rate = 0.0001
        weight_decay = 0.01
    elif NETWORK_TYPE == 'deep_residual':
        learning_rate = 0.0001
        weight_decay = 0.005
    elif NETWORK_TYPE in ['feature_attention', 'hybrid_advanced']:
        learning_rate = 0.00005
        weight_decay = 0.001
    elif NETWORK_TYPE == 'mega_transformer':
        learning_rate = 0.00002  # Very low LR for massive transformer
        weight_decay = 0.0001    # Minimal regularization for mega model
    elif NETWORK_TYPE == 'ultra_deep':
        learning_rate = 0.00003  # Low LR for ultra-deep networks
        weight_decay = 0.0005    # Light regularization
        
    action_sampler_optimizer = optim.AdamW(action_sampler_nn.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    optimal_strategy_optimizer = optim.AdamW(optimal_strategy_nn.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))

    num_iterations = args.iterations
    print(f"Starting MCCFR training for {num_iterations} iterations...")
    
    final_avg_strategies, exploitability_history, action_sampler_loss_history, optimal_strategy_loss_history = train_mccfr(num_iterations, kuhn_game)

    # Print exploitability summary
    if exploitability_history:
        print(f"\n--- Exploitability Tracking Summary ---")
        print(f"Initial exploitability (iter 100): {exploitability_history[0]:.6f}")
        print(f"Final exploitability: {exploitability_history[-1]:.6f}")
        print(f"Best exploitability achieved: {min(exploitability_history):.6f}")
        print(f"Exploitability reduction: {((exploitability_history[0] - exploitability_history[-1]) / exploitability_history[0] * 100):.2f}%")

    # Print neural network loss summary
    if action_sampler_loss_history:
        print(f"\n--- Neural Network Training Summary ---")
        print(f"Action Sampler - Initial loss: {action_sampler_loss_history[0]:.6f}, Final loss: {action_sampler_loss_history[-1]:.6f}")
        if optimal_strategy_loss_history:
            print(f"Optimal Strategy - Initial loss: {optimal_strategy_loss_history[0]:.6f}, Final loss: {optimal_strategy_loss_history[-1]:.6f}")

    # Plot training metrics
    if MATPLOTLIB_AVAILABLE:
        plot_training_metrics(exploitability_history, action_sampler_loss_history, optimal_strategy_loss_history)
    else:
        print("\nNote: Plotting disabled due to matplotlib compatibility issues.")
        print("You can still access the training data from the returned variables.")

    print("\n--- Final Average Strategies (Sample) ---")
    count = 0
    for infoset, strategy_vector in final_avg_strategies.items():
        if count >= 20 and len(final_avg_strategies) > 20 : # Print a limited sample
             print("... (and more strategies)")
             break

        player_card_str = infoset[0]
        history_str = "".join(infoset[1]) if infoset[1] else "Root" # Infoset[1] is tuple of action names
        
        strat_display_list = []
        has_strategy = False
        for action_idx, prob in enumerate(strategy_vector):
            if prob > 1e-4: # Only show actions with non-trivial probability
                strat_display_list.append(f"{IDX_TO_ACTION[action_idx].name}: {prob:.3f}")
                has_strategy = True
        
        if has_strategy: # Only print infosets where a strategy was computed
            print(f"Infoset: Card {player_card_str}, Hist: '{history_str}' -> Strategy: {{{', '.join(strat_display_list)}}}")
            count += 1
        elif not strat_display_list and np.sum(strategy_vector) < 1e-9 and info_sets[infoset]['visits'] > 0:
            # This might indicate an issue or an infoset where no strategy sum was accumulated meaningfully.
            # For Kuhn, all reachable decision infosets should have a strategy.
            pass # Suppress printing for zero-strategy entries unless debugging.

    if count == 0 and len(final_avg_strategies) > 0:
        print("No strategies with significant probabilities found to display (all probabilities are too small or zero).")
    elif len(final_avg_strategies) == 0:
        print("No infosets were generated.")

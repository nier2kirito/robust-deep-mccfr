"""
Neural network architectures for Deep MCCFR.

This module contains various neural network architectures designed for
learning strategies in imperfect information games, ranging from simple
feedforward networks to complex transformer-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaseNN(nn.Module):
    """
    Simple feedforward neural network with dropout regularization.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.dropout2 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.softmax(x)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and GELU activation.
    
    Args:
        hidden_size: Size of hidden layers
        dropout_rate: Dropout probability
    """
    
    def __init__(self, hidden_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Deep network with residual connections for better representation learning.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
        num_blocks: Number of residual blocks
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, num_blocks: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Multi-head attention mechanism.
    
    Args:
        hidden_size: Size of hidden layers
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout_rate: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    Network with self-attention mechanism to learn feature interactions.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
        num_heads: Number of attention heads
        num_layers: Number of attention layers
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, 
                 num_heads: int = 4, num_layers: int = 3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Reshape input to treat each feature as a token
        x = x.unsqueeze(-1)  # (batch, input_size, 1)
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
    """
    Hybrid architecture combining feature attention with deep residual processing.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Feature attention branch
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
        combined = torch.stack([x_attended_pooled, x_deep], dim=1)  # (batch, 2, hidden_size)
        fused, _ = self.fusion_attention(combined, combined, combined)
        fused = self.fusion_norm(fused)
        fused_flat = fused.view(batch_size, -1)  # (batch, 2 * hidden_size)
        
        # Output
        x = self.output_layers(fused_flat)
        return self.softmax(x)


class MegaTransformerNN(nn.Module):
    """
    Ultra-large transformer architecture with massive parameter count.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        intermediate_size: Size of intermediate layers in feedforward
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, 
                 num_heads: int = 32, num_layers: int = 12, intermediate_size: int = 8192):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Multi-scale input processing
        self.input_projections = nn.ModuleList([
            nn.Linear(input_size, hidden_size // 4),
            nn.Linear(input_size, hidden_size // 2),
            nn.Linear(input_size, hidden_size),
            nn.Linear(input_size, hidden_size * 2),
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(hidden_size * 4, hidden_size)
        self.scale_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, hidden_size) * 0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        self.layer_gates = nn.ParameterList()
        
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
            self.layer_gates.append(nn.Parameter(torch.ones(1)))
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(hidden_size, num_heads//2, batch_first=True)
        self.global_norm = nn.LayerNorm(hidden_size)
        
        # Output MLP
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Multi-scale input processing
        scale_outputs = []
        for i, proj in enumerate(self.input_projections):
            scale_out = proj(x)
            # Pad smaller scales to full hidden_size
            if i == 0:  # hidden_size//4
                scale_out = F.pad(scale_out, (0, self.hidden_size - self.hidden_size//4))
            elif i == 1:  # hidden_size//2  
                scale_out = F.pad(scale_out, (0, self.hidden_size - self.hidden_size//2))
            elif i == 3:  # hidden_size*2
                scale_out = scale_out[:, :self.hidden_size]  # Truncate
            scale_outputs.append(scale_out)
        
        # Fuse scales
        x_fused = torch.cat(scale_outputs, dim=-1)
        x = self.scale_fusion(x_fused)
        x = self.scale_norm(x)
        
        # Prepare for transformer
        x = x.unsqueeze(1).expand(-1, self.input_size, -1)
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
        x = pooled.squeeze(1)
        
        # Output
        x = self.output_mlp(x)
        return self.softmax(x)


class UltraDeepNN(nn.Module):
    """
    Ultra-deep network with bottleneck residual blocks and massive depth.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        num_actions: Number of possible actions
        num_blocks: Number of residual blocks
        bottleneck_factor: Factor for bottleneck compression
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_actions: int, 
                 num_blocks: int = 20, bottleneck_factor: int = 4):
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
        self.block_scales = nn.ParameterList()
        
        for i in range(num_blocks):
            # Varying bottleneck sizes for diversity
            current_bottleneck = self.bottleneck_size if i % 3 == 0 else self.bottleneck_size * 2
            
            block = nn.ModuleDict({
                'down_proj': nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ),
                'bottleneck': nn.Sequential(
                    nn.LayerNorm(current_bottleneck),
                    nn.Linear(current_bottleneck, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(current_bottleneck, current_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ),
                'up_proj': nn.Sequential(
                    nn.LayerNorm(current_bottleneck),
                    nn.Linear(current_bottleneck, hidden_size),
                    nn.Dropout(0.1)
                )
            })
            self.residual_blocks.append(block)
            self.block_scales.append(nn.Parameter(torch.ones(1) * 0.1))
        
        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            ),
            nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-path input processing
        path_outputs = []
        for path in self.input_paths:
            path_outputs.append(path(x))
        
        # Fuse paths
        x = torch.cat(path_outputs, dim=-1)
        x = self.path_fusion(x)
        
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
        x = torch.cat(scale_features, dim=-1)
        x = self.final_norm(x)
        
        # Output
        x = self.output_layers(x)
        return self.softmax(x)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_network(network_type: str, input_size: int, num_actions: int, **kwargs) -> nn.Module:
    """
    Factory function to create neural networks.
    
    Args:
        network_type: Type of network to create
        input_size: Size of input features
        num_actions: Number of possible actions
        **kwargs: Additional arguments for specific network types
        
    Returns:
        Neural network instance
    """
    
    network_configs = {
        'simple': {
            'class': BaseNN,
            'hidden_size': 512,
            'kwargs': {}
        },
        'deep_residual': {
            'class': DeepResidualNN,
            'hidden_size': 1024,
            'kwargs': {'num_blocks': 12}
        },
        'feature_attention': {
            'class': FeatureAttentionNN,
            'hidden_size': 1024,
            'kwargs': {'num_heads': 16, 'num_layers': 8}
        },
        'hybrid_advanced': {
            'class': HybridAdvancedNN,
            'hidden_size': 1024,
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
    
    if network_type not in network_configs:
        raise ValueError(f"Unknown network type: {network_type}")
    
    config = network_configs[network_type]
    # Merge provided kwargs with default kwargs
    merged_kwargs = {**config['kwargs'], **kwargs}
    
    return config['class'](
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_actions=num_actions,
        **merged_kwargs
    )

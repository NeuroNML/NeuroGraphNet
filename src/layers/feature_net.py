import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class FeatureAttention(nn.Module):
    """Attention mechanism for feature importance weighting"""
    def __init__(self, input_dim: int, attention_dim: Optional[int] = None):
        super().__init__()
        if attention_dim is None:
            attention_dim = input_dim // 2
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, input_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)  # (batch_size, 1)
        weighted_features = x * attention_weights  # (batch_size, input_dim)
        return weighted_features, attention_weights

class FeatureBlock(nn.Module):
    """A block of layers for feature processing with residual connection"""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)

class FeatureNet(nn.Module):
    """
    A deep neural network specifically designed for seizure detection using extracted features.
    Architecture:
    1. Feature attention layer to weight important features
    2. Multiple feature processing blocks with residual connections
    3. Final classification layer
    """
    def __init__(
        self,
        input_dim: int = 665,  # Number of input features (19 channels × 35 features)
        hidden_dims: List[int] = [512, 256, 128],  # Hidden layer dimensions
        dropout: float = 0.3,
        attention_dim: Optional[int] = None,
        num_classes: int = 1
    ):
        super().__init__()
        
        # Feature attention layer
        self.feature_attention = FeatureAttention(input_dim, attention_dim)
        
        # Feature processing blocks
        self.feature_blocks = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.feature_blocks.append(FeatureBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes),
            nn.BatchNorm1d(num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple containing:
            - logits: Model predictions of shape (batch_size, num_classes)
            - attention_weights: Feature attention weights of shape (batch_size, 1)
        """
        # Apply feature attention
        weighted_features, attention_weights = self.feature_attention(x)
        
        # Process through feature blocks
        features = weighted_features
        for block in self.feature_blocks:
            features = block(features)
        
        # Final classification
        logits = self.classifier(features)
        
        return logits, attention_weights
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the attention weights for feature importance analysis
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Feature importance weights of shape (batch_size, 1)
        """
        _, attention_weights = self.feature_attention(x)
        return attention_weights 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SimpleMLP(nn.Module):
    """
    A simple MLP model for testing dataset functionality.
    Architecture:
    1. Input layer
    2. Multiple hidden layers with ReLU and dropout
    3. Final classification layer
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        num_classes: int = 1
    ):
        super().__init__()
        
        # Calculate class weights for initialization
        # Assuming 80% negative, 20% positive
        neg_weight = 0.2  # weight for negative class
        pos_weight = 0.8  # weight for positive class
        
        layers = []
        prev_dim = input_dim
        
        # Input layer with batch norm
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights(neg_weight, pos_weight)
    
    def _init_weights(self, neg_weight: float, pos_weight: float):
        """Initialize weights with proper scaling and bias for class imbalance"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization with proper scaling
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    if m is self.classifier:
                        # Initialize classifier bias to account for class imbalance
                        # This helps prevent the model from predicting all zeros initially
                        nn.init.constant_(m.bias, torch.log(torch.tensor(pos_weight / neg_weight)))
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Model predictions of shape (batch_size, num_classes)
        """
        features = self.layers(x)
        return self.classifier(features) 
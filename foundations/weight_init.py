import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return torch.round(w, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return torch.round(w, decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Use torch.manual_seed(0) once at the start.
        # Sample all weights in layer order with torch.randn(out_dim, in_dim) * std:
        # first (hidden_dim, input_dim), then (hidden_dim, hidden_dim).
        # Keep weights unrounded; do not call the init helpers, which reset the seed.
        # Then sample x = torch.randn(1, input_dim) without resetting the seed.
        # Apply x = torch.relu(x @ W.T) at every layer, with no biases.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # Use round(x.std().item(), 2) with PyTorch's default sample std.
        torch.manual_seed(0)
        weights = []
        
        if init_type == 'xavier':
            std = math.sqrt(2 / (input_dim + hidden_dim))
        elif init_type == "kaiming":
            std = math.sqrt(2 / (input_dim))
        else:
            std = 1.0
        
        weights.append(torch.randn(hidden_dim, input_dim) * std)

        # Creat erest of the layers
        for i in range(num_layers - 1):
            if init_type == 'xavier':
                std = math.sqrt(2 / (hidden_dim + hidden_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2 / (hidden_dim))
            else:
                std = 1.0
            
            weights.append(torch.randn(hidden_dim, hidden_dim) * std)
        
        x = torch.randn(1, input_dim)
        desviations = []

        # Activations
        for w in weights:
            x = torch.relu(x @ w.T)
            desviations.append(round(x.std().item(), 2))

        return desviations




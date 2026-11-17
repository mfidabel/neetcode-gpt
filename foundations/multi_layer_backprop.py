import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        N = len(x)

        # Convert to numpy
        x = np.array(x)[np.newaxis, :]
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2) # n_2, n_1
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward pass
        z1 = x @ W1.T + b1 # n, n_1
        a1 = np.maximum(0, z1) # n, n_1
        z2 = a1 @ W2.T + b2 # n, n_2

        # Loss
        loss = (z2 - y_true) ** 2 # n, n_2
        
        # Backward
        dz2 = 2 * (z2 - y_true) # n, n_2

        # Params 2
        dW2 = dz2.T @ a1 # n_2, n_1
        db2 = dz2 # n, n_2

        # Activation 1
        da1 = dz2 @ W2 # n, n_1

        # Params 1
        dz1 = da1 * (a1 > 0).astype("int") # n, n_1
        dW1 = dz1.T @ x # n, n_0
        db1 = dz1 # n, n_1

        return {
            "loss": round(loss.mean(), 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1.mean(axis=0), 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2.mean(axis=0), 4)
        }



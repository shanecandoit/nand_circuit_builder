"""
CircuitBuilder: Learn logical circuits using NAND gates with continuous relaxation.

This implementation uses:
- DenseNet-style architecture (each gate connects to any previous gate/input)
- Straight-through estimator for gradient flow
- Temperature-annealed softmax for connection selection
- Quantile-based input discretization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class CircuitBuilder(nn.Module):
    """
    ML model that learns logical circuits from data using NAND gates.
    
    Parameters
    ----------
    n_gates : int, default=30
        Number of internal NAND gates to create
    input_buckets : int, default=5
        Number of quantile buckets per input feature
    temperature_init : float, default=2.0
        Initial temperature for softmax connection selection
    temperature_final : float, default=0.1
        Final temperature at end of training
    temperature_schedule : str, default='exponential'
        How temperature decreases: 'exponential', 'linear', or 'cosine'
    output_scaling : bool, default=True
        Whether to learn output scale multiplier (for unbounded regression)
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    l1_reg : float, default=0.001
        L1 regularization on connection weights for sparsity
    random_state : int or None, default=None
        Random seed for reproducibility
    device : str, default='cpu'
        Device to run computations on
    """
    
    def __init__(
        self,
        n_gates: int = 30,
        input_buckets: int = 5,
        temperature_init: float = 2.0,
        temperature_final: float = 0.1,
        temperature_schedule: str = 'exponential',
        output_scaling: bool = True,
        learning_rate: float = 0.01,
        l1_reg: float = 0.001,
        random_state: Optional[int] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.n_gates = n_gates
        self.input_buckets = input_buckets  # Keep for future use
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.temperature_schedule = temperature_schedule
        self.output_scaling = output_scaling
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.random_state = random_state
        self.device = device
        
        # Will be set during fit
        self.n_features_ = None
        self.n_outputs_ = None
        self.is_fitted_ = False
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
    
    def _get_temperature(self, epoch: int, max_epochs: int) -> float:
        """
        Continuous relaxation of NAND gate: nand(a, b) = 1 - (a * b)
        
        This allows gradient flow during training.
        """
        return 1.0 - (a * b)
    
    def _get_temperature(self, epoch: int, max_epochs: int) -> float:
        """Calculate temperature for current epoch."""
        progress = epoch / max_epochs
        
        if self.temperature_schedule == 'exponential':
            return self.temperature_init * (self.temperature_final / self.temperature_init) ** progress
        elif self.temperature_schedule == 'linear':
            return self.temperature_init - (self.temperature_init - self.temperature_final) * progress
        elif self.temperature_schedule == 'cosine':
            return self.temperature_final + 0.5 * (self.temperature_init - self.temperature_final) * \
                   (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown temperature schedule: {self.temperature_schedule}")
    
    def forward(
        self, 
        X: torch.Tensor, 
        temperature: float = 1.0,
        use_hard_selection: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the circuit.
        
        Parameters
        ----------
        X : tensor of shape (n_samples, n_features)
            Continuous input features
        temperature : float
            Temperature for softmax connection selection
        use_hard_selection : bool
            If True, use argmax (hard selection). If False, use soft weighted sum.
            Straight-through estimator uses hard forward, soft backward.
        
        Returns
        -------
        outputs : tensor of shape (n_samples, n_outputs)
            Final circuit outputs
        gate_outputs : list of tensors
            Outputs from each gate (for gradient computation)
        connection_probs : list of tensors
            Softmax probabilities for connections (for regularization)
        """
        n_samples = X.shape[0]
        
        # Available values: input features + constant 0 + constant 1 + previous gates
        # Start with inputs and constants
        available = torch.cat([
            X,
            torch.zeros((n_samples, 1), device=self.device),  # Constant 0
            torch.ones((n_samples, 1), device=self.device)     # Constant 1
        ], dim=1)
        
        gate_outputs = []
        connection_probs = []
        
        # Process each gate (DenseNet: each gate can use any previous)
        for gate_idx in range(self.n_gates):
            n_available = available.shape[1]
            
            # Get connection weights for this gate
            gate_weights = self.gate_weights[gate_idx, :n_available]
            
            # Softmax over available connections
            logits = gate_weights / temperature
            probs = torch.softmax(logits, dim=0)
            connection_probs.append(probs)
            
            # Select top 2 inputs for NAND gate
            top2_indices = torch.topk(probs, k=2).indices
            
            # Use the selected inputs
            input_a = available[:, top2_indices[0]]
            input_b = available[:, top2_indices[1]]
            
            # Apply NAND gate
            gate_output = self._nand_continuous(input_a, input_b)
            gate_outputs.append(gate_output)
            
            # Add this gate's output to available values (DenseNet)
            available = torch.cat([available, gate_output.reshape(-1, 1)], dim=1)
        
        # Output layer: weighted combination of all gates
        gate_matrix = torch.stack(gate_outputs, dim=1)  # (n_samples, n_gates)
        outputs = gate_matrix @ self.output_weights  # (n_samples, n_outputs)
        
        if self.output_scaling:
            outputs = outputs * self.output_scale
        
        return outputs, gate_outputs, connection_probs
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1000,
        verbose: bool = True,
        early_stopping_patience: int = 50
    ):
        """
        Fit the circuit to training data.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data
        y : array of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        early_stopping_patience : int
            Stop if no improvement for this many epochs
        
        Returns
        -------
        self : CircuitBuilder
            Fitted model
        """
        # Setup
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[1]
        
        # Convert to PyTorch tensors
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_torch = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Initialize parameters as nn.Parameter
        max_connections = n_features + 2 + self.n_gates  # features + const0 + const1 + gates
        self.gate_weights = nn.Parameter(
            torch.randn(self.n_gates, max_connections, device=self.device) * 0.1
        )
        self.output_weights = nn.Parameter(
            torch.randn(self.n_gates, self.n_outputs_, device=self.device) * 0.1
        )
        
        if self.output_scaling:
            self.output_scale = nn.Parameter(
                torch.ones(self.n_outputs_, device=self.device)
            )
        else:
            # Register as buffer so it's part of model state but not trained
            self.register_buffer('output_scale', torch.ones(self.n_outputs_, device=self.device))
        
        # Setup optimizer (after parameters are created)
        if self.output_scaling:
            params = [self.gate_weights, self.output_weights, self.output_scale]
        else:
            params = [self.gate_weights, self.output_weights]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            temperature = self._get_temperature(epoch, epochs)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, gate_outputs, connection_probs = self.forward(
                X_torch, temperature, use_hard_selection=False
            )
            
            # Compute loss (MSE)
            loss = torch.mean((outputs - y_torch) ** 2)
            
            # L1 regularization on connection weights
            l1_loss = self.l1_reg * sum(torch.sum(torch.abs(probs)) for probs in connection_probs)
            total_loss = loss + l1_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            # Logging
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f} | L1: {l1_loss.item():.6f} | Temp: {temperature:.3f}")
            
            # Early stopping
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted circuit.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        
        Returns
        -------
        predictions : array of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to PyTorch tensor
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Forward pass with hard selection (production mode)
        with torch.no_grad():
            outputs, _, _ = self.forward(X_torch, temperature=0.01, use_hard_selection=True)
        
        outputs_np = outputs.cpu().numpy()
        
        if self.n_outputs_ == 1:
            return outputs_np.ravel()
        return outputs_np
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score for regression tasks.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) or (n_samples, n_outputs)
        
        Returns
        -------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def prune(self, threshold: float = 0.01) -> 'CircuitBuilder':
        """
        Prune gates with negligible connections.
        
        Parameters
        ----------
        threshold : float
            Connection probability threshold for pruning
        
        Returns
        -------
        self : CircuitBuilder
            Pruned model
        """
        # TODO: Implement three-level pruning
        # 1. Remove unused input buckets
        # 2. Remove gates with low output contribution
        # 3. Remove redundant output heads
        
        print("Pruning not yet implemented")
        return self
    
    def to_binary_c_code(self) -> str:
        """
        Generate binary C code using bitwise NAND operations.
        
        Returns
        -------
        code : str
            C code implementing the circuit
        """
        # TODO: Implement code generation
        # Include quantile boundaries for input bucketization
        
        code = """
/* Generated circuit code */
#include <stdint.h>

// Quantile boundaries for input bucketization
// TODO: embed learned boundaries

uint8_t nand(uint8_t a, uint8_t b) {
    return ~(a & b) & 1;
}

// Circuit implementation
// TODO: generate gate network
"""
        return code
    
    def to_float_c_code(self) -> str:
        """
        Generate floating-point C code using continuous NAND.
        
        Returns
        -------
        code : str
            C code implementing the circuit
        """
        # TODO: Implement float code generation
        
        code = """
/* Generated circuit code (float version) */
#include <math.h>

float nand_float(float a, float b) {
    return 1.0f - (a * b);
}

// Circuit implementation
// TODO: generate gate network
"""
        return code

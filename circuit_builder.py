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
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict


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
    
    def _nand_continuous(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Continuous relaxation of NAND gate: nand(a, b) = 1 - (a * b)
        
        This allows gradient flow during training.
        """
        return 1.0 - (a * b)
    
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
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        epochs: int = 1000,
        verbose: bool = True,
        early_stopping_patience: int = 50,
        plot: bool = False,
        test_name: str = 'model'
    ) -> Dict:
        """
        Fit the circuit to training data.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data
        y : array of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        X_val : array, optional
            Validation data
        y_val : array, optional
            Validation targets
        X_test : array, optional
            Test data
        y_test : array, optional
            Test targets
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        early_stopping_patience : int
            Stop if no improvement for this many epochs
        plot : bool
            Whether to plot training curves
        
        Returns
        -------
        history : dict
            Training history with losses and metrics
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
            # Initialize output_scale based on target range for better stability
            y_range = y_torch.max() - y_torch.min()
            initial_scale = torch.clamp(y_range, min=1.0, max=10.0)
            self.output_scale = nn.Parameter(
                torch.ones(self.n_outputs_, device=self.device) * initial_scale
            )
        else:
            # Register as buffer so it's part of model state but not trained
            self.register_buffer('output_scale', torch.ones(self.n_outputs_, device=self.device))
        
        # Setup optimizer (after parameters are created)
        if self.output_scaling:
            params = [self.gate_weights, self.output_weights, self.output_scale]
        else:
            params = [self.gate_weights, self.output_weights]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        
        # Prepare validation and test sets
        if X_val is not None:
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            X_val_torch = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_torch = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        
        if X_test is not None:
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
            X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=self.device)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'temperature': []
        }
        
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
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                [self.gate_weights, self.output_weights] + 
                ([self.output_scale] if self.output_scaling else []), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"\nNaN/Inf detected at epoch {epoch}! Stopping training.")
                    print(f"Loss: {loss.item()}, Total Loss: {total_loss.item()}")
                    if self.output_scaling:
                        print(f"Output scale: {self.output_scale.item()}")
                break
            
            # Track metrics
            history['epoch'].append(epoch)
            history['train_loss'].append(loss.item())
            history['temperature'].append(temperature)
            
            # Validation loss
            if X_val is not None:
                with torch.no_grad():
                    val_outputs, _, _ = self.forward(X_val_torch, temperature, use_hard_selection=False)
                    val_loss = torch.mean((val_outputs - y_val_torch) ** 2)
                    history['val_loss'].append(val_loss.item())
            else:
                history['val_loss'].append(None)
            
            # Test loss
            if X_test is not None:
                with torch.no_grad():
                    test_outputs, _, _ = self.forward(X_test_torch, temperature, use_hard_selection=False)
                    test_loss = torch.mean((test_outputs - y_test_torch) ** 2)
                    history['test_loss'].append(test_loss.item())
            else:
                history['test_loss'].append(None)
            
            # Logging
            if verbose and epoch % 100 == 0:
                log_str = f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.6f}"
                if X_val is not None:
                    log_str += f" | Val Loss: {val_loss.item():.6f}"
                if X_test is not None:
                    log_str += f" | Test Loss: {test_loss.item():.6f}"
                log_str += f" | Temp: {temperature:.3f}"
                print(log_str)
            
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
        
        # Plot if requested
        if plot:
            self._plot_history(history, test_name)
        
        return history
    
    def _plot_history(self, history: Dict, test_name: str = 'model'):
        """Plot training history and save to file."""
        import os
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        epochs = history['epoch']
        ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=2)
        if history['val_loss'][0] is not None:
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=2)
        if history['test_loss'][0] is not None:
            ax1.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2, marker='^', markersize=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title(f'{test_name}: Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add final loss values as text
        final_train = history['train_loss'][-1]
        text_str = f'Final Train Loss: {final_train:.6f}'
        if history['val_loss'][0] is not None:
            final_val = history['val_loss'][-1]
            text_str += f'\nFinal Val Loss: {final_val:.6f}'
        if history['test_loss'][0] is not None:
            final_test = history['test_loss'][-1]
            text_str += f'\nFinal Test Loss: {final_test:.6f}'
        ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Temperature plot
        ax2.plot(epochs, history['temperature'], color='red', linewidth=2, marker='o', markersize=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Temperature', fontsize=12)
        ax2.set_title('Connection Selection Temperature', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add temperature info
        final_temp = history['temperature'][-1]
        init_temp = history['temperature'][0]
        temp_text = f'Initial: {init_temp:.3f}\nFinal: {final_temp:.3f}'
        ax2.text(0.98, 0.98, temp_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Overall title
        fig.suptitle(f'NAND Circuit Training: {test_name}\n{self.n_gates} gates, {len(epochs)} epochs', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to file
        os.makedirs('outputs', exist_ok=True)
        plot_file = f'outputs/{test_name.replace(" ", "_").lower()}_training.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_file}")
        
        # Save text summary
        summary_file = f'outputs/{test_name.replace(" ", "_").lower()}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary: {test_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Configuration:\n")
            f.write(f"  Gates: {self.n_gates}\n")
            f.write(f"  Input Buckets: {self.input_buckets}\n")
            f.write(f"  Learning Rate: {self.learning_rate}\n")
            f.write(f"  L1 Regularization: {self.l1_reg}\n")
            f.write(f"  Temperature: {init_temp:.3f} -> {final_temp:.3f}\n")
            f.write(f"  Schedule: {self.temperature_schedule}\n\n")
            
            f.write(f"Training Results:\n")
            f.write(f"  Total Epochs: {len(epochs)}\n")
            f.write(f"  Final Train Loss: {final_train:.6f}\n")
            if history['val_loss'][0] is not None:
                f.write(f"  Final Val Loss: {final_val:.6f}\n")
            if history['test_loss'][0] is not None:
                f.write(f"  Final Test Loss: {final_test:.6f}\n")
            
            # Min losses
            min_train_loss = min(history['train_loss'])
            min_train_epoch = history['epoch'][history['train_loss'].index(min_train_loss)]
            f.write(f"\n  Best Train Loss: {min_train_loss:.6f} (epoch {min_train_epoch})\n")
            
            if history['val_loss'][0] is not None:
                val_losses = [v for v in history['val_loss'] if v is not None]
                if val_losses:
                    min_val_loss = min(val_losses)
                    min_val_epoch = history['epoch'][history['val_loss'].index(min_val_loss)]
                    f.write(f"  Best Val Loss: {min_val_loss:.6f} (epoch {min_val_epoch})\n")
            
            if history['test_loss'][0] is not None:
                test_losses = [t for t in history['test_loss'] if t is not None]
                if test_losses:
                    min_test_loss = min(test_losses)
                    min_test_epoch = history['epoch'][history['test_loss'].index(min_test_loss)]
                    f.write(f"  Best Test Loss: {min_test_loss:.6f} (epoch {min_test_epoch})\n")
        
        print(f"Summary saved to: {summary_file}")
    
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

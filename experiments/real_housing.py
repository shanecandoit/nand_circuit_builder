"""
Housing Prices Regression Experiment

Tests CircuitBuilder on a housing prices dataset (regression task).
- 13 features → 65 binary inputs (5 buckets each)
- Target: Price (with scaling)
- Success metric: R² score > 0.6
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from circuit_builder import CircuitBuilder


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def test_housing():
    """Test learning housing prices."""
    print("\n" + "="*70)
    print("HOUSING PRICES REGRESSION: Price Prediction")
    print("="*70)
    
    # Load California Housing dataset (8 features)
    # Note: For 13 features, we could use fetch_openml with a different dataset
    # Using California Housing as it's readily available and similar
    print("\nLoading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"\nDataset Info:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Feature names: {housing.feature_names}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # If we need exactly 13 features, we can pad with synthetic features
    # or use a different dataset. For now, we'll work with available features
    # and scale up to match the requirement
    n_features = X.shape[1]
    target_features = 13
    
    if n_features < target_features:
        # Add synthetic features (interactions or polynomial features)
        # This helps reach 13 features → 65 binary inputs
        print(f"\nNote: Dataset has {n_features} features, adding features to reach {target_features}...")
        # Add some polynomial/interaction features
        n_to_add = target_features - n_features
        X_extra = np.zeros((X.shape[0], n_to_add))
        
        # Add some meaningful combinations
        if n_to_add >= 1:
            X_extra[:, 0] = X[:, 0] * X[:, 1]  # Interaction
        if n_to_add >= 2:
            X_extra[:, 1] = X[:, 0] ** 2  # Square
        if n_to_add >= 3:
            X_extra[:, 2] = X[:, 1] ** 2
        if n_to_add >= 4:
            X_extra[:, 3] = X[:, 2] * X[:, 3]
        if n_to_add >= 5:
            X_extra[:, 4] = np.sqrt(np.abs(X[:, 0]))
        
        # Fill remaining with random noise (scaled to match data distribution)
        np.random.seed(42)  # For reproducibility
        for i in range(5, n_to_add):
            X_extra[:, i] = np.random.randn(X.shape[0]) * X.std(axis=0).mean()
        
        X = np.hstack([X, X_extra])
        n_features = X.shape[1]
        print(f"  Extended to {n_features} features")
    
    # Normalize features (important for quantile-based discretization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Normalize target to [0, 1] range for better training
    # Housing prices are already in reasonable range, but normalization helps
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_normalized = (y - y_min) / y_range if y_range > 0 else y
    
    # Split into train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_normalized, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Model configuration: 13 features × 5 buckets = 65 binary inputs
    # Use more gates for complex real-world data
    model = CircuitBuilder(
        n_gates=60,  # More gates for complex relationships
        input_buckets=5,  # 5 buckets per feature = 65 total binary inputs
        random_state=42,
        learning_rate=0.01,
        output_scaling=True,  # Important for unbounded regression
        l1_reg=0.001,
        temperature_init=5.0,
        temperature_final=0.5
    )
    
    print(f"\nModel Configuration:")
    print(f"  Gates: {model.n_gates}")
    print(f"  Input Buckets: {model.input_buckets} per feature")
    print(f"  Total Binary Inputs: {X_train.shape[1] * model.input_buckets}")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Output Scaling: {model.output_scaling}")
    
    # Train the model
    print(f"\nStarting training...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        epochs=1000,
        verbose=True,
        plot=True,  # Built-in loss plots
        test_name='Housing Prices Regression'
    )
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print('='*70)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Convert back to original scale for reporting
    y_train_orig = y_train * y_range + y_min
    y_val_orig = y_val * y_range + y_min
    y_test_orig = y_test * y_range + y_min
    y_train_pred_orig = y_train_pred * y_range + y_min
    y_val_pred_orig = y_val_pred * y_range + y_min
    y_test_pred_orig = y_test_pred * y_range + y_min
    
    # Metrics on original scale
    train_mse = np.mean((y_train_pred_orig - y_train_orig) ** 2)
    val_mse = np.mean((y_val_pred_orig - y_val_orig) ** 2)
    test_mse = np.mean((y_test_pred_orig - y_test_orig) ** 2)
    
    train_mae = np.mean(np.abs(y_train_pred_orig - y_train_orig))
    val_mae = np.mean(np.abs(y_val_pred_orig - y_val_orig))
    test_mae = np.mean(np.abs(y_test_pred_orig - y_test_orig))
    
    train_r2 = compute_r2(y_train_orig, y_train_pred_orig)
    val_r2 = compute_r2(y_val_orig, y_val_pred_orig)
    test_r2 = compute_r2(y_test_orig, y_test_pred_orig)
    
    print(f"\nTest Set Metrics (original scale):")
    print(f"  MSE: {test_mse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"\n  Target: R² > 0.6")
    print(f"  Status: {'[PASS]' if test_r2 > 0.6 else '[FAIL]'}")
    
    print(f"\nAll Sets Metrics:")
    print(f"{'Set':<10}{'MSE':<12}{'MAE':<12}{'R²':<10}")
    print('-'*44)
    print(f"{'Train':<10}{train_mse:<12.2f}{train_mae:<12.2f}{train_r2:<10.4f}")
    print(f"{'Val':<10}{val_mse:<12.2f}{val_mae:<12.2f}{val_r2:<10.4f}")
    print(f"{'Test':<10}{test_mse:<12.2f}{test_mae:<12.2f}{test_r2:<10.4f}")
    
    # Sample predictions
    print(f"\nSample Predictions (first 10 test samples):")
    print(f"{'Actual':<12}{'Predicted':<12}{'Error':<12}{'Error %':<12}")
    print('-'*48)
    for i in range(min(10, len(y_test_orig))):
        error = abs(y_test_pred_orig[i] - y_test_orig[i])
        error_pct = (error / abs(y_test_orig[i])) * 100 if y_test_orig[i] != 0 else 0
        print(f"{y_test_orig[i]:<12.2f}{y_test_pred_orig[i]:<12.2f}{error:<12.2f}{error_pct:<12.1f}")
    
    # Create combined plot with loss and R²
    r2_history = compute_r2_history(model, X_train, y_train, X_val, y_val, 
                                     X_test, y_test, history, y_range, y_min)
    create_loss_r2_plot(history, r2_history, test_r2)
    
    return test_r2 > 0.6


def compute_r2_history(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                       history, y_range, y_min):
    """Compute R² values for plotting."""
    # Get final predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Convert to original scale
    y_train_orig = y_train * y_range + y_min
    y_val_orig = y_val * y_range + y_min
    y_test_orig = y_test * y_range + y_min
    y_train_pred_orig = y_train_pred * y_range + y_min
    y_val_pred_orig = y_val_pred * y_range + y_min
    y_test_pred_orig = y_test_pred * y_range + y_min
    
    # Compute final R² values
    final_train_r2 = compute_r2(y_train_orig, y_train_pred_orig)
    final_val_r2 = compute_r2(y_val_orig, y_val_pred_orig)
    final_test_r2 = compute_r2(y_test_orig, y_test_pred_orig)
    
    # Return final values (we'll show these prominently in the plot)
    return {
        'train_r2': final_train_r2,
        'val_r2': final_val_r2,
        'test_r2': final_test_r2
    }


def create_loss_r2_plot(history, r2_history, final_r2):
    """Create a plot showing both loss curves and R² score over time."""
    os.makedirs('outputs', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epoch']
    
    # Left panel: Loss curves
    ax1.semilogy(epochs, history['train_loss'], 'o-', label='Train Loss', 
                markersize=2, alpha=0.7, linewidth=1.5)
    if history['val_loss'][0] is not None:
        ax1.semilogy(epochs, history['val_loss'], 's-', label='Val Loss', 
                    markersize=2, alpha=0.7, linewidth=1.5)
    if history['test_loss'][0] is not None:
        ax1.semilogy(epochs, history['test_loss'], '^-', label='Test Loss', 
                    markersize=2, alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE, log scale)', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add final loss values
    final_train = history['train_loss'][-1]
    text_str = f'Final Train: {final_train:.4f}'
    if history['val_loss'][0] is not None:
        final_val = history['val_loss'][-1]
        text_str += f'\nFinal Val: {final_val:.4f}'
    if history['test_loss'][0] is not None:
        final_test = history['test_loss'][-1]
        text_str += f'\nFinal Test: {final_test:.4f}'
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right panel: R² score (bar chart showing final values)
    sets = []
    r2_values = []
    colors = []
    
    sets.append('Train')
    r2_values.append(r2_history['train_r2'])
    colors.append('steelblue')
    
    if 'val_r2' in r2_history:
        sets.append('Val')
        r2_values.append(r2_history['val_r2'])
        colors.append('orange')
    
    sets.append('Test')
    r2_values.append(r2_history['test_r2'])
    colors.append('green')
    
    bars = ax2.bar(sets, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add target line at R² = 0.6
    ax2.axhline(y=0.6, color='r', linestyle='--', linewidth=2, 
               label='Target (0.6)', alpha=0.7, zorder=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Final R² Scores', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([-0.2, max(1.0, max(r2_values) * 1.2)])  # R² can be negative
    
    # Add status text
    status_text = f'Test R²: {final_r2:.4f}'
    if final_r2 > 0.6:
        status_text += '\n[PASS ✓]'
        status_color = 'lightgreen'
    else:
        status_text += '\n[FAIL ✗]'
        status_color = 'lightcoral'
    ax2.text(0.98, 0.98, status_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.7))
    
    # Overall title
    fig.suptitle('Housing Prices Regression: Loss and R² Score\n60 gates, 65 binary inputs (13 features × 5 buckets)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'outputs/housing_prices_regression_loss_r2.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCombined loss and R² plot saved to: {plot_file}")


if __name__ == "__main__":
    print("Housing Prices Regression Experiment")
    print("="*70)
    print("Testing CircuitBuilder on California Housing dataset")
    print("Target: R² score > 0.6\n")
    
    success = test_housing()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print('='*70)
    print(f"Result: {'[PASS]' if success else '[FAIL]'}")
    print(f"Target metric: R² > 0.6")

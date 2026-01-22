"""
Experiment: Circle Area with Different Gate Counts
Compare performance with 16, 32, and 64 gates
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from circuit_builder import CircuitBuilder


def run_experiment():
    """Run circle area test with different gate counts."""
    print("=" * 70)
    print("EXPERIMENT: Circle Area with Different Gate Counts")
    print("=" * 70)
    
    # Prepare data
    np.random.seed(42)
    X = np.random.rand(400, 1) * 2
    y = np.pi * (X[:, 0] ** 2)
    
    print(f"\nData: r in [0, 2], area in [0, {np.pi * 4:.2f}]")
    
    # Split into train/val/test
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:300], y[200:300]
    X_test, y_test = X[300:], y[300:]
    
    # Test different gate counts
    gate_counts = [16, 32, 64, 128]
    results = {}
    
    for n_gates in gate_counts:
        print(f"\n{'=' * 70}")
        print(f"Testing with {n_gates} gates")
        print('=' * 70)
        
        model = CircuitBuilder(
            n_gates=n_gates, 
            input_buckets=7, 
            random_state=42,
            learning_rate=0.01, 
            output_scaling=True,
            temperature_init=5.0,
            temperature_final=0.5
        )
        
        history = model.fit(
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test,
            epochs=500, 
            verbose=True, 
            plot=False,  # We'll make our own comparison plot
            test_name=f'Circle Area {n_gates} Gates'
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        pct_error = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
        
        results[n_gates] = {
            'history': history,
            'predictions': y_pred,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'pct_error': pct_error
        }
        
        print(f"\n{n_gates} Gates Results:")
        print(f"  Test MSE: {mse:.6f}")
        print(f"  Test MAE: {mae:.6f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  Test % Error: {pct_error:.2f}%")
    
    # Create comparison plots
    create_comparison_plots(results, X_test, y_test)
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print('=' * 70)
    print(f"{'Gates':<10}{'MSE':<12}{'MAE':<12}{'R²':<10}{'% Error':<10}")
    print('-' * 70)
    for n_gates in gate_counts:
        r = results[n_gates]
        print(f"{n_gates:<10}{r['mse']:<12.6f}{r['mae']:<12.6f}{r['r2']:<10.4f}{r['pct_error']:<10.2f}")
    
    return results


def create_comparison_plots(results, X_test, y_test):
    """Create comparison plots for different gate counts."""
    gate_counts = sorted(results.keys())
    n_gates_count = len(gate_counts)
    
    # Create figure with 2 rows, dynamic columns based on number of gate counts
    # Use at least 2 columns, but expand if needed
    n_cols = max(2, n_gates_count)
    fig = plt.figure(figsize=(5 * n_cols, 10))
    
    # Row 1: Training curves for each gate count
    for idx, n_gates in enumerate(gate_counts):
        ax = plt.subplot(2, n_cols, idx + 1)
        history = results[n_gates]['history']
        
        epochs = history['epoch']
        ax.semilogy(epochs, history['train_loss'], 'o-', label='Train', 
                   markersize=3, alpha=0.7)
        ax.semilogy(epochs, history['val_loss'], 's-', label='Val', 
                   markersize=3, alpha=0.7)
        ax.semilogy(epochs, history['test_loss'], '^-', label='Test', 
                   markersize=3, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title(f'{n_gates} Gates - Training Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final loss as text
        final_test = history['test_loss'][-1]
        ax.text(0.98, 0.02, f'Final Test Loss: {final_test:.4f}',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Prediction scatter plots
    for idx, n_gates in enumerate(gate_counts):
        ax = plt.subplot(2, n_cols, n_cols + idx + 1)
        y_pred = results[n_gates]['predictions']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect', linewidth=2)
        
        ax.set_xlabel('Actual Area')
        ax.set_ylabel('Predicted Area')
        ax.set_title(f'{n_gates} Gates - Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² as text
        r2 = results[n_gates]['r2']
        pct_err = results[n_gates]['pct_error']
        ax.text(0.02, 0.98, f'R² = {r2:.4f}\n% Error = {pct_err:.2f}%',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs/experiments', exist_ok=True)
    plot_file = 'outputs/experiments/circle_area_gate_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {plot_file}")
    
    # Create summary comparison plot (all losses on one plot)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: All training curves
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    for idx, n_gates in enumerate(gate_counts):
        history = results[n_gates]['history']
        epochs = history['epoch']
        color = colors[idx % len(colors)]  # Cycle through colors if needed
        ax1.semilogy(epochs, history['test_loss'], 
                    label=f'{n_gates} gates', 
                    color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Loss (log scale)')
    ax1.set_title('Test Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Bar chart of final metrics
    metrics = ['MSE', 'MAE', '% Error']
    x_pos = np.arange(len(gate_counts))
    width = 0.25
    
    mse_values = [results[g]['mse'] for g in gate_counts]
    mae_values = [results[g]['mae'] for g in gate_counts]
    pct_values = [results[g]['pct_error'] for g in gate_counts]
    
    # Normalize for visualization
    max_mse = max(mse_values)
    max_mae = max(mae_values)
    max_pct = max(pct_values)
    
    ax2.bar(x_pos - width, [m/max_mse for m in mse_values], width, 
           label='MSE (norm)', color='blue', alpha=0.7)
    ax2.bar(x_pos, [m/max_mae for m in mae_values], width, 
           label='MAE (norm)', color='orange', alpha=0.7)
    ax2.bar(x_pos + width, [p/max_pct for p in pct_values], width, 
           label='% Error (norm)', color='green', alpha=0.7)
    
    ax2.set_xlabel('Number of Gates')
    ax2.set_ylabel('Normalized Metric Value')
    ax2.set_title('Final Metrics Comparison (Normalized)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(g) for g in gate_counts])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    summary_file = 'outputs/experiments/circle_area_gate_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"📊 Summary plot saved to: {summary_file}")
    
    plt.close('all')


if __name__ == '__main__':
    results = run_experiment()
    print(f"\n✅ Experiment complete!")

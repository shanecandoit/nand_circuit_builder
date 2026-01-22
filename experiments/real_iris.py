"""
Iris Classification Experiment

Tests CircuitBuilder on the sklearn Iris dataset (classification task).
- 4 features → 20 binary inputs (5 buckets each)
- 3 classes → 3 output heads
- Success metric: Accuracy > 90%
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from circuit_builder import CircuitBuilder


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def test_iris():
    """Test learning iris species classification."""
    print("\n" + "="*70)
    print("IRIS CLASSIFICATION: Species Prediction")
    print("="*70)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"\nDataset Info:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Feature names: {iris.feature_names}")
    print(f"  Classes: {len(iris.target_names)}")
    print(f"  Class names: {iris.target_names}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Normalize features (important for quantile-based discretization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode targets for multi-class classification
    n_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Split into train/val/test (60/20/20)
    X_train, X_temp, y_train_oh, y_temp_oh = train_test_split(
        X_scaled, y_onehot, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val_oh, y_test_oh = train_test_split(
        X_temp, y_temp_oh, test_size=0.5, random_state=42, stratify=y_temp_oh.argmax(axis=1)
    )
    
    # Get class labels for metrics
    y_train = y_train_oh.argmax(axis=1)
    y_val = y_val_oh.argmax(axis=1)
    y_test = y_test_oh.argmax(axis=1)
    
    print(f"\nData Split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Model configuration: 4 features × 5 buckets = 20 binary inputs
    # 3 classes → 3 output heads (auto-detected from y shape)
    model = CircuitBuilder(
        n_gates=40,  # More gates for classification
        input_buckets=5,  # 5 buckets per feature = 20 total binary inputs
        random_state=42,
        learning_rate=0.01,
        output_scaling=False,  # No scaling needed for classification
        l1_reg=0.001,
        temperature_init=2.0,
        temperature_final=0.1
    )
    
    print(f"\nModel Configuration:")
    print(f"  Gates: {model.n_gates}")
    print(f"  Input Buckets: {model.input_buckets} per feature")
    print(f"  Total Binary Inputs: {X_train.shape[1] * model.input_buckets}")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Task: Classification (auto-detected)")
    
    # Train the model
    print(f"\nStarting training...")
    history = model.fit(
        X_train, y_train_oh,
        X_val, y_val_oh,
        X_test, y_test_oh,
        epochs=1000,
        verbose=True,
        plot=True,  # Built-in loss plots
        test_name='Iris Classification'
    )
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    # Predictions (get class probabilities and argmax for class labels)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Convert predictions to class labels
    train_pred_class = train_pred.argmax(axis=1)
    val_pred_class = val_pred.argmax(axis=1)
    test_pred_class = test_pred.argmax(axis=1)
    
    # Calculate accuracies
    train_acc = compute_accuracy(y_train, train_pred_class)
    val_acc = compute_accuracy(y_val, val_pred_class)
    test_acc = compute_accuracy(y_test, test_pred_class)
    
    print(f"\nAccuracy Results:")
    print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Success criteria
    success = test_acc >= 0.90
    print(f"\n{'='*70}")
    print(f"SUCCESS CRITERION: Test Accuracy >= 90%")
    print(f"Result: {'✓ PASSED' if success else '✗ FAILED'} (Test Acc = {test_acc*100:.2f}%)")
    print(f"{'='*70}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(iris.target_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = compute_accuracy(y_test[class_mask], test_pred_class[class_mask])
            print(f"  {class_name:12s}: {class_acc:.4f} ({class_acc*100:.2f}%) [{class_mask.sum()} samples]")
    
    # Confusion Matrix (simple text format)
    print(f"\nConfusion Matrix (Test Set):")
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_test)):
        confusion[y_test[i], test_pred_class[i]] += 1
    
    print("           " + "".join(f"{iris.target_names[i][:8]:>10s}" for i in range(n_classes)))
    for i in range(n_classes):
        print(f"{iris.target_names[i][:8]:10s} " + "".join(f"{confusion[i, j]:10d}" for j in range(n_classes)))
    
    # Plot confusion matrix
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(confusion, cmap='Blues', interpolation='nearest')
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(iris.target_names)
        ax.set_yticklabels(iris.target_names)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Iris Classification Confusion Matrix (Test Set)')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax.text(j, i, str(confusion[i, j]),
                             ha="center", va="center", color="black" if confusion[i, j] < confusion.max()/2 else "white")
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'experiments')
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, 'iris_confusion_matrix.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {plot_file}")
        plt.close()
    
    # Circuit Statistics
    print(f"\n{'='*70}")
    print("CIRCUIT STATISTICS")
    print(f"{'='*70}")
    
    # Active connections analysis
    n_possible_inputs = X_train.shape[1] * model.input_buckets + model.n_gates + 2  # inputs + gates + constants
    print(f"  Total possible inputs per gate: {n_possible_inputs}")
    print(f"  Total gates: {model.n_gates}")
    print(f"  Output heads: {model.n_outputs_}")
    
    # Count active connections (threshold at 0.01)
    active_threshold = 0.01
    if hasattr(model, 'gate_weights') and model.gate_weights is not None:
        # Apply softmax to get probabilities (convert to numpy)
        gate_weights_np = model.gate_weights.detach().cpu().numpy()
        gate_probs = np.exp(gate_weights_np - gate_weights_np.max(axis=1, keepdims=True))
        gate_probs = gate_probs / gate_probs.sum(axis=1, keepdims=True)
        
        active_connections = (gate_probs > active_threshold).sum()
        total_connections = gate_probs.size
        sparsity = 1 - (active_connections / total_connections)
        
        print(f"  Active connections (>{active_threshold}): {active_connections}/{total_connections}")
        print(f"  Sparsity: {sparsity:.2%}")
    
    # Temperature schedule final value
    if hasattr(model, 'temperature'):
        print(f"  Final temperature: {model.temperature:.4f}")
    
    # Save results summary
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'iris_classification_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("IRIS CLASSIFICATION EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Dataset:\n")
        f.write(f"  Features: {X.shape[1]}\n")
        f.write(f"  Samples: {X.shape[0]}\n")
        f.write(f"  Classes: {len(iris.target_names)}\n")
        f.write(f"  Class names: {', '.join(iris.target_names)}\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Gates: {model.n_gates}\n")
        f.write(f"  Input Buckets: {model.input_buckets}\n")
        f.write(f"  Binary Inputs: {X_train.shape[1] * model.input_buckets}\n")
        f.write(f"  Output Heads: {model.n_outputs_}\n")
        f.write(f"  Learning Rate: {model.learning_rate}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
        f.write(f"  Val Accuracy:   {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        f.write(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        
        f.write(f"Success Criterion: Test Accuracy >= 90%\n")
        f.write(f"Status: {'PASSED' if success else 'FAILED'}\n\n")
        
        f.write("Per-Class Performance (Test Set):\n")
        for i, class_name in enumerate(iris.target_names):
            class_mask = y_test == i
            if class_mask.sum() > 0:
                class_acc = compute_accuracy(y_test[class_mask], test_pred_class[class_mask])
                f.write(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%) [{class_mask.sum()} samples]\n")
        
        f.write("\nConfusion Matrix (Test Set):\n")
        f.write("           " + "".join(f"{iris.target_names[i][:8]:>10s}" for i in range(n_classes)) + "\n")
        for i in range(n_classes):
            f.write(f"{iris.target_names[i][:8]:10s} " + "".join(f"{confusion[i, j]:10d}" for j in range(n_classes)) + "\n")
    
    print(f"\nResults summary saved to: {summary_file}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'success': success,
        'model': model,
        'history': history
    }


if __name__ == '__main__':
    results = test_iris()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Final Test Accuracy: {results['test_acc']*100:.2f}%")
    print(f"Target: 90%")
    print(f"Status: {'✓ SUCCESS' if results['success'] else '✗ NEEDS IMPROVEMENT'}")

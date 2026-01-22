import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from circuit_builder import CircuitBuilder


def test_average():
    """Test learning the average of two inputs."""
    print("\n" + "="*60)
    print("TEST 1: Average Function - f(x, y) = (x + y) / 2")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(400, 2)
    y = (X[:, 0] + X[:, 1]) / 2
    
    # Split into train/val/test
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:300], y[200:300]
    X_test, y_test = X[300:], y[300:]
    
    model = CircuitBuilder(n_gates=15, input_buckets=5, random_state=42, learning_rate=0.05)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test, 
                       epochs=500, verbose=True, plot=True, test_name='Average Function')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: ({X_test[i, 0]:.3f}, {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01


def test_rectangle_area():
    """Test learning rectangle area = width * height."""
    print("\n" + "="*60)
    print("TEST 2: Rectangle Area - f(w, h) = w * h")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(400, 2)
    y = X[:, 0] * X[:, 1]
    
    # Split into train/val/test
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:300], y[200:300]
    X_test, y_test = X[300:], y[300:]
    
    model = CircuitBuilder(n_gates=20, input_buckets=5, random_state=42, learning_rate=0.05)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=500, verbose=True, plot=True, test_name='Rectangle Area')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  ({X_test[i, 0]:.3f} x {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01


def test_circle_area():
    """Test learning circle area = π * r²."""
    print("\n" + "="*60)
    print("TEST 3: Circle Area - f(r) = pi * r^2")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(400, 1) * 2
    y = np.pi * (X[:, 0] ** 2)
    
    # Split into train/val/test
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:300], y[200:300]
    X_test, y_test = X[300:], y[300:]
    
    model = CircuitBuilder(n_gates=25, input_buckets=7, random_state=42, 
                          learning_rate=0.01, output_scaling=True)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=1_000, verbose=True, plot=True, test_name='Circle Area')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    pct_error = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test % Error: {pct_error:.2f}%")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        error_pct = abs((y_pred[i] - y_test[i]) / y_test[i]) * 100
        print(f"  r={X_test[i, 0]:.3f} -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f} ({error_pct:.1f}% error)")
    
    return pct_error < 10


def test_sigmoid():
    """Test learning sigmoid approximation."""
    print("\n" + "="*60)
    print("TEST 4: Sigmoid Approximation - f(x) ~= 1 / (1 + exp(-x))")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(300, 1) * 2
    y = 1 / (1 + np.exp(-X[:, 0]))
    
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test, y_test = X[250:], y[250:]
    
    model = CircuitBuilder(n_gates=20, input_buckets=7, random_state=42, learning_rate=0.05)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=500, verbose=True, plot=True, test_name='Sigmoid Approximation')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: {X_test[i, 0]:.3f} -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.05


def test_step_function():
    """Test learning a step function."""
    print("\n" + "="*60)
    print("TEST 5: Step Function - f(x) = 1 if x > 0.5 else 0")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(300, 1)
    y = (X[:, 0] > 0.5).astype(float)
    
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test, y_test = X[250:], y[250:]
    
    model = CircuitBuilder(n_gates=10, input_buckets=5, random_state=42, learning_rate=0.05)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=500, verbose=True, plot=True, test_name='Step Function')
    
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(float)
    accuracy = np.mean(y_pred_binary == y_test)
    
    print(f"\nTest Accuracy: {accuracy:.3f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: {X_test[i, 0]:.3f} -> Pred: {y_pred[i]:.3f} ({y_pred_binary[i]:.0f}), Actual: {y_test[i]:.0f}")
    
    return accuracy > 0.85


def test_multiply():
    """Test learning multiplication."""
    print("\n" + "="*60)
    print("TEST 6: Multiplication - f(x, y) = x * y")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(300, 2)
    y = X[:, 0] * X[:, 1]
    
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:250], y[200:250]
    X_test, y_test = X[250:], y[250:]
    
    model = CircuitBuilder(n_gates=20, input_buckets=5, random_state=42, learning_rate=0.05)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=500, verbose=True, plot=True, test_name='Multiplication')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: ({X_test[i, 0]:.3f}, {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: ({X_test[i, 0]:.3f}, {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01


def test_circle_area():
    """Test learning circle area = π * r²."""
    print("\n" + "="*60)
    print("TEST 6: Circle Area - f(r) = pi * r^2")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.rand(400, 1) * 2  # radius in [0, 2]
    y = np.pi * (X[:, 0] ** 2)  # area
    
    # Split into train/val/test
    X_train, y_train = X[:200], y[:200]
    X_val, y_val = X[200:300], y[200:300]
    X_test, y_test = X[300:], y[300:]
    
    model = CircuitBuilder(n_gates=25, input_buckets=7, random_state=42, 
                          learning_rate=0.05, output_scaling=True)
    history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=500, verbose=True, plot=True, test_name='Circle Area')
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    pct_error = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test % Error: {pct_error:.2f}%")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: r={X_test[i, 0]:.3f} -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return pct_error < 10  # Less than 10% error


if __name__ == "__main__":
    print("CircuitBuilder Demo Tests")
    print("Testing NAND gate circuits with PyTorch backprop\n")
    
    results = {}
    
    # Run tests
    results['average'] = test_average()
    results['rectangle_area'] = test_rectangle_area()
    results['circle_area'] = test_circle_area()
    results['sigmoid'] = test_sigmoid()
    results['step'] = test_step_function()
    results['multiply'] = test_multiply()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

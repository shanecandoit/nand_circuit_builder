import numpy as np
from circuit_builder import CircuitBuilder


def test_average():
    """Test learning the average of two inputs."""
    print("\n" + "="*60)
    print("TEST 1: Average Function - f(x, y) = (x + y) / 2")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.rand(200, 2)
    y_train = (X_train[:, 0] + X_train[:, 1]) / 2
    
    X_test = np.random.rand(50, 2)
    y_test = (X_test[:, 0] + X_test[:, 1]) / 2
    
    model = CircuitBuilder(n_gates=15, input_buckets=5, random_state=42, learning_rate=0.05)
    model.fit(X_train, y_train, epochs=500, verbose=True)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: ({X_test[i, 0]:.3f}, {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01


def test_sigmoid():
    """Test learning sigmoid approximation."""
    print("\n" + "="*60)
    print("TEST 2: Sigmoid Approximation - f(x) ≈ 1 / (1 + exp(-x))")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.randn(200, 1) * 2
    y_train = 1 / (1 + np.exp(-X_train[:, 0]))
    
    X_test = np.random.randn(50, 1) * 2
    y_test = 1 / (1 + np.exp(-X_test[:, 0]))
    
    model = CircuitBuilder(n_gates=20, input_buckets=7, random_state=42, learning_rate=0.05)
    model.fit(X_train, y_train, epochs=500, verbose=True)
    
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
    print("TEST 3: Step Function - f(x) = 1 if x > 0.5 else 0")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.rand(200, 1)
    y_train = (X_train[:, 0] > 0.5).astype(float)
    
    X_test = np.random.rand(50, 1)
    y_test = (X_test[:, 0] > 0.5).astype(float)
    
    model = CircuitBuilder(n_gates=10, input_buckets=5, random_state=42, learning_rate=0.05)
    model.fit(X_train, y_train, epochs=500, verbose=True)
    
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
    print("TEST 4: Multiplication - f(x, y) = x * y")
    print("="*60)
    
    np.random.seed(42)
    X_train = np.random.rand(200, 2)
    y_train = X_train[:, 0] * X_train[:, 1]
    
    X_test = np.random.rand(50, 2)
    y_test = X_test[:, 0] * X_test[:, 1]
    
    model = CircuitBuilder(n_gates=20, input_buckets=5, random_state=42, learning_rate=0.05)
    model.fit(X_train, y_train, epochs=500, verbose=True)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Input: ({X_test[i, 0]:.3f}, {X_test[i, 1]:.3f}) -> Pred: {y_pred[i]:.3f}, Actual: {y_test[i]:.3f}")
    
    return mse < 0.01


if __name__ == "__main__":
    print("CircuitBuilder Demo Tests")
    print("Testing NAND gate circuits with PyTorch backprop\n")
    
    results = {}
    
    # Run tests
    results['average'] = test_average()
    results['sigmoid'] = test_sigmoid()
    results['step'] = test_step_function()
    results['multiply'] = test_multiply()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:15s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

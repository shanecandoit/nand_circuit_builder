
# CircuitBuilder ML Project Plan

## Project Overview

Build a novel ML system that learns logical circuits using NAND gates with continuous relaxation, capable of training with backpropagation and compiling to efficient binary logic or floating-point implementations.

## Core Architecture

### 1. CircuitBuilder Class Design

**Key Components:**

- **Input Layer**: Quantile-based discretization converting continuous features into binary buckets (quantile boundaries saved for C code generation)
- **Constant Inputs**: Gates can select constant 0 or 1 as inputs (NAND primarily needs 1)
- **Hidden Gates**: DenseNet-style architecture - each gate can connect to ANY previous input/gate output (not strictly layered)
- **Output Layer**: NAND gate heads with optional scaling for different tasks
- **Connection Learning**: Softmax-based input selection with temperature annealing
- **Gradient Flow**: Straight-through estimator enables differentiation through entire circuit

**API Design (scikit-learn compatible):**

```python
class CircuitBuilder:
    def __init__(self, n_waves=3, gates_per_wave=10, input_buckets=5, 
                 temperature_schedule='exponential', output_scaling=True)
    def fit(X, y, epochs=1000)
    def predict(X)
    def score(X, y)
    def prune(threshold=0.01)
    def to_binary_c_code()
    def to_float_c_code()
```

### 2. NAND Gate Implementation

**Continuous NAND Function:**

- Training: `nand(a, b) = 1 - (a * b)`
- Binary: `nand(a, b) = ~(a & b)`

**Gate Structure:**

- Each gate has learnable connection weights to ALL previous gates/inputs (DenseNet-style)
- Available inputs: all input buckets, all previous gate outputs, constant 0, constant 1
- Softmax over connections with temperature parameter (exactly 2 inputs selected per gate)
- Temperature annealing: starts high (soft connections) → decreases (hard selection)
- Straight-through estimator: forward pass uses hard selection, backward pass uses soft gradients

### 3. Input Quantization Strategy

**Per-feature bucketization:**

- Compute quantiles on training data (e.g., 5 buckets = [0, 0.25, 0.5, 0.75, 1.0])
- **Store quantile boundaries** for later use in C code generation
- Convert continuous value to one-hot encoding across buckets
- Each bucket becomes a binary input node
- Example: 2 features × 5 buckets = 10 binary inputs
- Quantiles tracked during training and embedded in compiled code

### 4. Connection Selection Mechanism

**Gumbel-Softmax or Temperature-scaled Softmax:**

- Each gate maintains weight vector over all possible inputs
- Sample/select top-2 inputs using `softmax(logits / temperature)`
- Temperature schedule: `T(epoch) = T_init * decay^epoch`
- Gradually forces selection to concentrate on best 2 inputs

### 5. Output Layer Design

**Task-specific configurations:**

- **Regression (0-1)**: Single head, no scaling
- **Regression (unbounded)**: Single/multiple heads with learnable scale multiplier
- **Classification**: Multiple heads (one per class) - all outputs share early gates (DenseNet efficiency)
- Multi-output tasks benefit from shared gate infrastructure

**Loss functions:**

- Regression: MSE
- Classification: Cross-entropy
- Regularization: L1 on connection weights to encourage sparsity

### 6. Pruning Strategy

**Three-level pruning:**

1. **Input nodes**: Remove buckets never strongly selected
2. **Internal gates**: Remove gates with negligible contribution to outputs
3. **Output heads**: Remove heads with low gradients or redundant predictions

**Criteria:**

- Connection weight threshold (< 0.01 after softmax)
- Gradient magnitude over final N epochs
- Output variance contribution

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

- [ ] Implement NANDGate class with continuous function
- [ ] Implement input quantization/bucketization
- [ ] Build basic forward pass through gate layers
- [ ] Implement temperature-scheduled softmax selection

### Phase 2: Training Loop (Week 1-2)

- [ ] Backpropagation through NAND operations
- [ ] Connection weight optimization
- [ ] Temperature annealing schedule
- [ ] Early stopping and convergence monitoring

### Phase 3: Pruning & Optimization (Week 2)

- [ ] Implement three-level pruning
- [ ] Add regularization for sparsity
- [ ] Post-training optimization pass
- [ ] Binary connection hardening

### Phase 4: Code Generation (Week 2-3)

- [ ] Circuit graph extraction
- [ ] Binary C code generator (bitwise logic)
- [ ] Float C code generator (continuous NAND)
- [ ] Code optimization (dead code elimination)

### Phase 5: Testing & Experiments (Week 3-4)

- [ ] Unit tests for all components
- [ ] Synthetic function experiments
- [ ] Real dataset experiments
- [ ] Performance benchmarking

## Experimental Progression

### Level 1: Simple Functions

1. **Sigmoid approximation**: `σ(x) ≈ circuit(x)`
2. **Tanh approximation**: `tanh(x) ≈ circuit(x)`
3. **Increment**: `f(x) = x + 1`
4. **Step function**: `f(x) = 1 if x > 0.5 else 0`

**Success criteria**: < 1% error on test set

### Level 2: Geometric Functions

1. **Rectangle area**: `area(x, y) = x * y`
2. **Circle area**: `area(r) = π * r²`

**Success criteria**: < 5% error, circuit uses < 20 gates

### Level 3: Real ML Datasets

**Diabetes (Regression):**

- 10 features → 50 binary inputs (5 buckets each)
- Target: Disease progression (continuous)
- Metric: R² score > 0.4

**Iris (Classification):**

- 4 features → 20 binary inputs
- 3 classes → 3 output heads
- Metric: Accuracy > 90%

**Housing Prices (Regression):**

- 13 features → 65 binary inputs
- Target: Price (with scaling)
- Metric: R² score > 0.6

## Technical Challenges & Solutions

### Challenge 1: Vanishing Gradients

**Solution**: Careful initialization, gradient clipping, skip connections between waves

### Challenge 2: Connection Selection Convergence

**Solution**: Two-phase training - explore (high T) then exploit (low T)

### Challenge 3: Output Scaling Discovery

**Solution**: Learn log-scale multiplier, separate learning rate for scale vs. gates

### Challenge 4: Binary vs. Continuous Mismatch

**Solution**: Straight-through estimator throughout training - hard binary selection in forward pass, soft gradients in backward pass. Ensures compiled circuit matches training behavior.

### Challenge 5: Circuit Efficiency

**Solution**: Aggressive pruning, gate reuse detection, common subexpression elimination

## Code Structure

```txt
circuit_ml/
├── core/
│   ├── nand_gate.py          # NAND gate implementation
│   ├── input_layer.py         # Quantization logic
│   ├── connection_selector.py # Softmax selection
│   └── output_head.py         # Output layer variants
├── model/
│   ├── circuit_builder.py     # Main model class
│   ├── trainer.py             # Training loop
│   └── pruner.py              # Pruning algorithms
├── codegen/
│   ├── binary_gen.py          # Binary C code
│   ├── float_gen.py           # Float C code
│   └── optimizer.py           # Code optimization
├── experiments/
│   ├── synthetic.py           # Simple function tests
│   ├── sklearn_datasets.py    # Real dataset experiments
│   └── benchmarks.py          # Performance testing
└── tests/
    └── ...                    # Unit tests
```

## Success Metrics

1. **Learning**: Can approximate target functions with < 10% test error
2. **Efficiency**: Pruned circuits use < 50% of initial gates
3. **Convergence**: Connection selection reaches binary (entropy < 0.1) by end of training
4. **Compilation**: Generated C code produces identical outputs to Python model
5. **Performance**: Binary C code runs 100x+ faster than float version

## Extensions for Future Work

- Multi-wave skip connections
- Gate type diversity (NOR, XOR as alternatives)
- Hierarchical quantization (adaptive bucket counts)
- Circuit visualization and interpretation
- Hardware synthesis for FPGA/ASIC deployment

## Conclusion

?

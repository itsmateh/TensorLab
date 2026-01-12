# Tensor Zero

A Rust-based tensor operations library built from scratch. This project aims to provide fundamental tensor manipulations and mathematical operations for numerical computing.

## Project Status

This library is in early development. Currently implementing core tensor functionality and basic operations.

## Current Features

- **Tensor Creation**: Initialize tensors with data and shape validation
- **Element Access**: Get values using row-major indexing
- **Basic Operations**: 
  - Element-wise addition with operator overloading (`+`)
- **Error Handling**: Comprehensive error types for shape mismatches and invalid operations

## Implementation Details

**Core Structure**
```rust
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

The library uses a flat `Vec<f32>` for data storage with row-major ordering, making it memory-efficient and cache-friendly.

### Supported Operations

**Tensor Creation**
```rust
let data = vec![1.0, 2.0, 3.0, 4.0];
let shape = vec![2, 2];
let tensor = Tensor::new(&data, &shape).unwrap();
```

**Element Access**
```rust
let value = tensor.get_value(&[0, 1]).unwrap(); // Get element at row 0, col 1
```

**Addition**
```rust
let tensor_a = Tensor::new(&data, &shape).unwrap();
let tensor_b = Tensor::new(&data, &shape).unwrap();
let result = (&tensor_a + &tensor_b).unwrap(); // Operator overloading
```

## 🎯 Roadmap

- [ ] Additional arithmetic operations (subtraction, multiplication, division)
- [ ] Matrix multiplication
- [ ] Broadcasting support
- [ ] Multi-dimensional indexing
- [ ] Reshape operations
- [ ] Transposition
- [ ] Reduction operations (sum, mean, max, min)
- [ ] GPU acceleration
- [ ] Automatic differentiation


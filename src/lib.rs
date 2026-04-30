//! TensorLab: A library for tensor operations
//!
//! This library provides a Tensor type with support for common linear algebra operations
//! including element-wise operations and matrix multiplication.

mod error;
mod operations;
mod tensor;

// Re-export public API
pub use error::Error;
pub use tensor::Tensor;

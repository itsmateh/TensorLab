use crate::error::Error;
use crate::tensor::Tensor;
use std::ops::{Add, Mul, Sub};

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor, Error>;

    fn add(self, other_tensor: &'b Tensor) -> Self::Output {
        Tensor::add(self, other_tensor)
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor, Error>;

    fn sub(self, other_tensor: &'b Tensor) -> Self::Output {
        Tensor::sub(self, other_tensor)
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Result<Tensor, Error>;

    fn mul(self, other_tensor: &'b Tensor) -> Self::Output {
        Tensor::mul(self, other_tensor)
    }
}

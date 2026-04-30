use crate::error::Error;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: &[f32], shape: &[usize]) -> Result<Tensor, Error> {
        let req_values: usize = shape.iter().product();
        if data.len() != req_values {
            return Err(Error::NumbersOfElementsError);
        }
        Ok(Tensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
        })
    }

    pub fn get_value(&self, coords: &[usize]) -> Result<f32, Error> {
        let shape_size = self.shape.len();
        if coords.len() != shape_size {
            return Err(Error::PositionValueError);
        }
        let row = coords[0];
        let col = coords[1];
        if row >= self.shape[0] || col >= self.shape[1] {
            return Err(Error::ShapeMismatchError);
        }
        let width = self.shape[1];
        let index = (row * width) + col;
        Ok(self.data[index])
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn add(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data: Vec<f32> = self
            .data
            .iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor {
            data: new_tensor_data,
            shape: self.shape.clone(),
        })
    }

    pub fn sub(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data: Vec<f32> = self
            .data
            .iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Tensor {
            data: new_tensor_data,
            shape: self.shape.clone(),
        })
    }

    /// Hadamard / element-wise product
    pub fn mul(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }
        let new_tensor_data: Vec<f32> = self
            .data
            .iter()
            .zip(other_tensor.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor {
            data: new_tensor_data,
            shape: self.shape.clone(),
        })
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let new_tensor_data: Vec<f32> = self.data.iter().map(|a| a * scalar).collect();
        Tensor {
            data: new_tensor_data,
            shape: self.shape.clone(),
        }
    }

    pub fn matmul(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        // Validate shapes: self must be (n, k) and other must be (k, m)
        if (self.shape[1] != other_tensor.shape[0])
            || (self.shape.len() != 2 || other_tensor.shape.len() != 2)
        {
            return Err(Error::ShapeMismatchError);
        }

        let n = self.shape[0]; // A rows
        let m = other_tensor.shape[1]; // B columns
        let k = self.shape[1]; // Common dimension (A_cols / B_rows)

        let mut new_tensor_data: Vec<f32> = vec![0.0; n * m];

        // ans[n][m] = sum(A[n][k], B[k][m])
        // Row-major indexing: index = row * total_cols + col
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for c in 0..k {
                    let a = self.data[(i * k) + c];
                    let b = other_tensor.data[j + (m * c)];
                    sum += a * b;
                }
                new_tensor_data[(i * m) + j] = sum;
            }
        }

        Ok(Tensor {
            data: new_tensor_data,
            shape: vec![n, m],
        })
    }
}

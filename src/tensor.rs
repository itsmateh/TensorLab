use std::mem::swap;

use rand::rand_core::block;

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

    fn get_index(&self, row: usize, col: usize) -> usize {
        row * self.shape[1] + col
    }

    fn get_transpose_index(&self, row: usize, col: usize) -> usize {
        col * self.shape[0] + row
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

    pub fn transpose_naive(&self) -> Result<Tensor, Error> {
        if self.shape.len() != 2 {
            return Err(Error::ShapeMismatchError);
        }

        let n_rows = self.shape[0];
        let n_cols = self.shape[1];

        let mut transposed_data = vec![0.0; n_rows * n_cols];

        for i in 0..n_rows {
            for j in 0..n_cols {
                let original_index = self.get_index(i, j);
                // (i, j) moves to (j, i)
                let transposed_index = self.get_transpose_index(i, j);
                transposed_data[transposed_index] = self.data[original_index];
            }
        }

        Ok(Tensor {
            data: transposed_data,
            shape: vec![n_cols, n_rows], // shape inverted
        })
    }

    // only square matrix
    pub fn transpose_inplace(&mut self) -> Result<(), Error>{
        let n = self.shape[0]; let m = self.shape[1];
        if n != m || self.shape.len() != 2 {
            return Err(Error::ShapeMismatchError)
        }
        // swap (i,j) -> (j,i)
        for i in 0..n{
            for j in (i+1)..n{
                let idx1 = self.get_index(i,j);
                let idx2 = self.get_transpose_index(j,i);
                self.data.swap(idx1, idx2);
            }       
        }

        Ok(())
    }

    // block for n*n matrix
    pub fn transpose_blocked(&self, block_size:usize) -> Result<Tensor, Error>{
        if self.shape.len() != 2 {
            return Err(Error::ShapeMismatchError)
        }
        let n:usize = self.shape[0];
        let mut transposed_data = vec![0.0; n*n];
        // processing block
        for block_i in (0..n).step_by(block_size){
            for block_j in (0..n).step_by(block_size){
                // process data inside block
                for i in block_i..std::cmp::min(block_size+block_i, n){
                    for j in block_j..std::cmp::min(block_size+block_j, n){
                        let idx1=self.get_index(i, j);
                        let idx2=self.get_transpose_index(i, j);
                        transposed_data[idx2] = self.data[idx1];
                    }
                }

            }
        } 
        return Ok(Tensor { data:transposed_data, shape:vec![n,n] });
    }
    
    pub fn find_optimal_block(&self) -> usize{
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return 64;
        }
        let n = self.shape[0];
        let block_sizes=vec![16, 32, 64, 128, 256];
        let iterations = match n {
            256..=1024 => 50,
            1025..=2048 => 20,
            2049..=4096 => 5,
            _ => 2,
        };

        let mut best_block_size = 64;
        let mut best_time = f64::INFINITY;

        for block_size in block_sizes {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = self.transpose_blocked(block_size);
            }
            let elapsed = start.elapsed().as_secs_f64() / iterations as f64;

            if elapsed < best_time {
                best_time = elapsed;    
                best_block_size = block_size;
            }
        }

        best_block_size
    }
    
    pub fn matmul(&self, other_tensor: &Tensor) -> Result<Tensor, Error> {
        if (self.shape[1] != other_tensor.shape[0])
            || (self.shape.len() != 2 || other_tensor.shape.len() != 2)
        {
            return Err(Error::ShapeMismatchError);
        }

        let n = self.shape[0]; // A rows
        let m = other_tensor.shape[1]; // B columns
        let k = self.shape[1]; // common dimension (A_cols / B_rows)

        let mut new_tensor_data: Vec<f32> = vec![0.0; n * m];

        // ans[n][m] = sum(A[n][k], B[k][m])
        // row-major: index = row * total_cols + col
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

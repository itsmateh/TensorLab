use std::fmt;
use std::ops::Add;
#[derive(Debug, Clone)]
enum Error{
    NumbersOfElementsError,
    PositionValueError,
    ShapeMismatchError
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
       match *self {
           Error::NumbersOfElementsError => write!(f, "Error: elements mismatch shape"),
           Error::PositionValueError => write!(f, "Error: index out of bounds"),
           Error::ShapeMismatchError => write!(f, "Error: shapes don't match"),
       }
    }
}

#[derive(Debug, Clone)]
struct Tensor{
    data:Vec<f32>,
    shape:Vec<usize>,
}
impl<'a,'b> Add<&'b Tensor> for &'a Tensor{
    type Output = Result<Tensor, Error>;
    fn add(self, other_tensor:&'b Tensor) -> Self::Output{
        Tensor::add(self, other_tensor)
    }
}
impl Tensor{
    fn new(data:&[f32], shape:&[usize]) -> Result<Tensor, Error> {
        let req_values=shape.iter().product();
        if data.len() != req_values {
            return Err(Error::NumbersOfElementsError);
        }
        return Ok(Tensor{ 
            data: data.to_vec(), 
            shape: shape.to_vec(),
        });
    }

    // row major [row, col]
    fn get_value(&self, coords:&[usize]) -> Result<f32, Error> {
        let shape_size=self.shape.len();
        let coords_size=coords.len();
        if shape_size != coords_size {
            return Err(Error::PositionValueError);
        }
        
        let row=coords[0];
        let col=coords[1];
        if row >= self.shape[0] || col >= self.shape[1]{
            return Err(Error::ShapeMismatchError);
        }

        let width = self.shape[1];
        let index=(row*width)+col;
        return Ok(self.data[index]);

    }

    fn add(&self, other_tensor:&Tensor) -> Result<Tensor, Error>{
        if self.shape != other_tensor.shape {
            return Err(Error::ShapeMismatchError);
        }   

        // let mut new_tensor_data = Vec::with_capacity(self.data.len());
        // for i in 0..self.data.len() {
        //     let sum=self.data[i]+other_tensor.data[i];
        //     new_tensor_data.push(sum);
        // }

        // functional like
        let new_tensor_data:Vec<f32> = self.data.iter()
            .zip(other_tensor.data.iter())
            .map(|(tensor_A, tensor_B)| tensor_A + tensor_B)
            .collect();

        let row=self.shape[0];
        let col=self.shape[1];
        return Ok(Tensor { data: new_tensor_data, shape:self.shape.clone() });
    }
}
fn main() {
    println!("tensor zero, el proyecto de operaciones tensoriales!");

    let data=vec![1.0, 2.0, 3.0, 4.0];
    let shape=vec![2,2]; 

    let tensor_1=Tensor::new(&data, &shape).unwrap();
    let tensor_2=Tensor::new(&data, &shape).unwrap();
    // let tensor_3 = Tensor::add(&tensor_1, &tensor_2).unwrap();
    let tensor_3 = (&tensor_1+&tensor_2).unwrap();
    println!("A: {:?} + B {:?} = {:?}", tensor_1.data, tensor_2.data, tensor_3.data);
}

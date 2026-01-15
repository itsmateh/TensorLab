use tensor_zero::{Tensor, Error};

fn create_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(&data, &shape).unwrap()
}

#[test]
fn test_initialization() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(&data, &shape);
    assert!(tensor.is_ok());
}

#[test]
fn test_initialization_error() {
    let data = vec![1.0, 2.0, 3.0]; 
    let shape = vec![2, 2];
    let tensor = Tensor::new(&data, &shape);
    assert!(tensor.is_err());
}

#[test]
fn test_add() {
    let t1 = create_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = create_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = (&t1 + &t2).unwrap();
    assert_eq!(result.data(), &vec![6.0, 8.0, 10.0, 12.0]);
    assert_eq!(result.shape(), &vec![2, 2]);
}

#[test]
fn test_sub() {
    let t1 = create_tensor(vec![10.0, 20.0], vec![1, 2]);
    let t2 = create_tensor(vec![1.0, 2.0], vec![1, 2]);
    let result = (&t1 - &t2).unwrap();
    assert_eq!(result.data(), &vec![9.0, 18.0]);
}

#[test]
fn test_mul_element_wise() {
    let t1 = create_tensor(vec![2.0, 3.0], vec![1, 2]);
    let t2 = create_tensor(vec![4.0, 5.0], vec![1, 2]);
    let result = (&t1 * &t2).unwrap();
    assert_eq!(result.data(), &vec![8.0, 15.0]);
}

#[test]
fn test_mul_scalar() {
    let t1 = create_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let scalar = 10.0;
    let result = t1.mul_scalar(scalar);
    assert_eq!(result.data(), &vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_matmul_shapes() {
    let t1 = create_tensor(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let t2 = create_tensor(vec![7., 8., 9., 1., 2., 3.], vec![3, 2]);
    let result = t1.matmul(&t2).unwrap();
    assert_eq!(result.shape(), &vec![2, 2]);
    assert_eq!(result.data(), &vec![31.0, 19.0, 85.0, 55.0]);
}

#[test]
fn test_matmul_invalid_shapes() {
    let t1 = create_tensor(vec![1., 1., 1., 1., 1., 1.], vec![2, 3]);
    let t2 = create_tensor(vec![1., 1., 1., 1., 1., 1.], vec![2, 3]); 
    let result = t1.matmul(&t2);
    assert!(result.is_err());
}
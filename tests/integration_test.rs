use tensorlab::Tensor;

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

// ===== Add
#[test]
fn test_add() {
    let t1 = create_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = create_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = (&t1 + &t2).unwrap();
    assert_eq!(result.data(), &vec![6.0, 8.0, 10.0, 12.0]);
    assert_eq!(result.shape(), &vec![2, 2]);
}

// ===== Sub

#[test]
fn test_sub() {
    let t1 = create_tensor(vec![10.0, 20.0], vec![1, 2]);
    let t2 = create_tensor(vec![1.0, 2.0], vec![1, 2]);
    let result = (&t1 - &t2).unwrap();
    assert_eq!(result.data(), &vec![9.0, 18.0]);
}

// ===== Mul

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

// ===== Transpose

#[test]
fn test_matmul_identity() {
    let a = create_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);    
    let identity = create_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let res = a.matmul(&identity).unwrap();
    assert_eq!(res.data(), a.data());
    assert_eq!(res.shape(), a.shape());
}

#[test]
fn test_matmul_associativity() {
    let a = create_tensor(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = create_tensor(vec![2., 0., 1., 2.], vec![2, 2]);
    let c = create_tensor(vec![1., 1., 0., 1.], vec![2, 2]);
    let ab = a.matmul(&b).unwrap();
    let abc_1 = ab.matmul(&c).unwrap();
    let bc = b.matmul(&c).unwrap();
    let abc_2 = a.matmul(&bc).unwrap();
    assert_eq!(abc_1.data(), abc_2.data());
}

#[test]
fn test_matmul_neural_network_layer() {
    let input = create_tensor(vec![1., 2., 3.], vec![1, 3]);
    let weights = create_tensor(vec![1., 4., 2., 5., 3., 6.], vec![3, 2]);
    let output = input.matmul(&weights).unwrap();
    assert_eq!(output.shape(), &vec![1, 2]);
    assert_eq!(output.data(), &vec![14.0, 32.0]);
}

#[test]
fn test_matmul_zeros() {
    let a = create_tensor(vec![1., 2., 3., 4.], vec![2, 2]);
    let zeros = create_tensor(vec![0., 0., 0., 0.], vec![2, 2]);
    let res = a.matmul(&zeros).unwrap();
    assert_eq!(res.data(), &vec![0., 0., 0., 0.]);
}


#[test]
fn test_matmul_rectangular_complex() {
    let a = create_tensor(vec![1.; 8], vec![2, 4]);
    let b = create_tensor(vec![2.; 12], vec![4, 3]);
    let res = a.matmul(&b).unwrap();
    assert_eq!(res.shape(), &vec![2, 3]);
    assert_eq!(res.data(), &vec![8.; 6]);
}

// ===== Transpose

#[test]
fn test_transpose_rectangular(){
    let tens = create_tensor(vec![1.,3.,5.,2.,4.,6.], vec![2,3]);
    let tens_trans = tens.transpose_naive().unwrap();
    assert_eq!(tens_trans.shape(), &vec![3,2]);
    assert_eq!(tens_trans.data(), &vec![1.,2.,3.,4.,5.,6.]);
}

#[test]
fn test_transpose_identity() {
    let t = create_tensor(vec![1., 0., 0., 1.], vec![2, 2]);
    let t_t = t.transpose_naive().unwrap();
    assert_eq!(t_t.data(), t.data());
    assert_eq!(t_t.shape(), t.shape());
}

#[test]
fn test_transpose_vector_row_to_col() {
    let t = create_tensor(vec![1., 2., 3., 4.], vec![1, 4]);
    let t_t = t.transpose_naive().unwrap();
    assert_eq!(t_t.shape(), &vec![4, 1]);
    assert_eq!(t_t.data(), &vec![1., 2., 3., 4.]);
}

#[test]
fn test_transpose_double() {
    let t = create_tensor(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let t_final = t.transpose_naive().unwrap().transpose_naive().unwrap();
    assert_eq!(t_final.data(), t.data());
    assert_eq!(t_final.shape(), t.shape());
}

#[test]
fn test_transpose_symmetry() {
    let t = create_tensor(vec![1., 2., 3., 4.], vec![2, 2]);
    let t_t = t.transpose_naive().unwrap();
    assert_eq!(t_t.data()[1], 3.0);
    assert_eq!(t_t.data()[2], 2.0);
}


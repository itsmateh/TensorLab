use tensor_zero::Tensor;
fn main(){
    println!("tensor zero, the tensor operations project!");

    // let data=vec![1.0, 2.0, 3.0, 4.0];
    // let shape=vec![2,2]; 

    // let tensor_1=Tensor::new(&data, &shape).unwrap();
    // let tensor_2=Tensor::new(&data, &shape).unwrap();
    // // let tensor_3 = Tensor::add(&tensor_1, &tensor_2).unwrap();
    // let tensor_3 = (&tensor_1+&tensor_2).unwrap();
    // let tensor_4 = (&tensor_1-&tensor_2).unwrap(); 
    // let tensor_5 = (&tensor_1*&tensor_2).unwrap(); 

    
    // // println!(" A: {:?} \n + \n B: {:?} \n ------------------------ \n C: {:?}", tensor_1.data(), tensor_2.data(), tensor_3.data());
    // // println!("####################################");
    // // println!(" A: {:?} \n - \n B: {:?} \n ------------------------ \n C: {:?}", tensor_1.data(), tensor_2.data(), tensor_4.data());
    // // println!("####################################");
    // // println!(" A: {:?} \n * \n B: {:?} \n ------------------------ \n C: {:?}", tensor_1.data(), tensor_2.data(), tensor_5.data());

    // let a = Tensor::new(&vec![2.0, 4.0, 1.0, 0.0], &vec![2, 2]).unwrap();
    // let b = Tensor::new(&vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &vec![2, 3]).unwrap();
    // let c = a.matmul(&b).unwrap();
    // println!("MatMul: {:?}", c.data()); 
    // // expected answer [10.0, 22.0, 34.0, 1.0, 3.0, 5.0]

}


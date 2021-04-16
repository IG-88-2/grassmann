use std::cmp::min;
use crate::{Number, core::matrix::Matrix};



pub fn trace<T: Number>(A: &Matrix<T>) -> T {

    let mut acc = T::from_f64(0.).unwrap();
    let d = min(A.rows, A.columns);
    
    for i in 0..d {
        acc += A[[i, i]]; 
    }

    acc
}

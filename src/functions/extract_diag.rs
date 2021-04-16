use crate::{Number, core::vector::Vector, core::matrix::Matrix};



pub fn extract_diag<T: Number>(A: &Matrix<T>) -> Vector<T> {

    let zero = T::from_f64(0.).unwrap();

    let mut v = vec![zero; A.rows];

    for i in 0..A.rows {
        v[i] = A[[i, i]];
    } 

    Vector::new(v)
}
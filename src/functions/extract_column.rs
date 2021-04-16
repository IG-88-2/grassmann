use crate::{Number, core::vector::Vector, core::matrix::Matrix};



pub fn extract_column<T: Number>(A: &Matrix<T>, i: usize) -> Vector<T> {

    let zero = T::from_f64(0.).unwrap();

    let mut c = vec![zero; A.rows]; 

    for j in 0..A.rows {
        c[j] = A[[j, i]];
    }

    Vector::new(c)
}
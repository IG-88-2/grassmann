use crate::{Number, core::vector::Vector, core::matrix::Matrix};



pub fn exchange_rows<T: Number>(A: &mut Matrix<T>, i: usize, j: usize) {

    for k in 0..A.columns {
        let t = A[[i, k]];
        A[[i, k]] = A[[j, k]];
        A[[j, k]] = t;
    }
}
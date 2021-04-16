use crate::{Number, core::vector::Vector, core::matrix::Matrix};



pub fn exchange_columns<T: Number>(A: &mut Matrix<T>, i: usize, j: usize) {

    for k in 0..A.rows {
        let t = A[[k, i]];
        A[[k, i]] = A[[k, j]];
        A[[k, j]] = t;
    }
}

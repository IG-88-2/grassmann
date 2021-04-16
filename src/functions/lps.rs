use crate::{Number, core::matrix::Matrix};



pub fn lps<T: Number>(M: &Matrix<T>, n: usize) -> Matrix<T> {
    let mut A = Matrix::new(n, n);

    for i in 0..n {
        for j in 0..n {
            A[[i,j]] = M[[i,j]];
        }
    }

    A
}
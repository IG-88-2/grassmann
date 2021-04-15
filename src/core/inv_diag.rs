use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use crate::{matrix, Number};



pub fn inv_diag<T: Number>(A: &Matrix<T>) -> Matrix<T> {

    assert!(A.is_diag(), "inv_diag matrix should be diagonal");

    assert!(A.is_square(), "inv_diag matrix should be square");

    let one = T::from_f64(1.).unwrap();
    
    let mut A_inv: Matrix<T> = Matrix::new(A.rows, A.columns);

    for i in 0..A.rows {
        A_inv[[i, i]] = one / A[[i, i]];
    }

    A_inv
}

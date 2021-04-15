use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_upper_triangular, vector::Vector};
use crate::{matrix, Number};



pub fn inv_upper_triangular<T: Number>(A: &Matrix<T>) -> Option<Matrix<T>> {

    //assert!(self.is_upper_triangular(), "matrix should be upper triangular");

    if A.rows != A.columns {
        return None;
    }

    let id: Matrix<T> = Matrix::id(A.rows);

    let bs = id.into_basis();

    let mut list: Vec<Vector<T>> = Vec::new();

    for i in 0..bs.len() {
        let b = &bs[i];
        let b_inv = solve_upper_triangular(A, b);
        if b_inv.is_none() {
            return None;
        }
        list.push(b_inv.unwrap());
    }

    let A_inv = Matrix::from_basis(list);

    Some(
        A_inv
    )
}
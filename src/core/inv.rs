use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, vector::Vector};
use crate::{matrix, Number};



pub fn inv<T: Number>(A: &Matrix<T>, lu: &lu<T>) -> Option<Matrix<T>> {
        
    if A.rows != A.columns {
       return None;
    }

    if !lu.d.is_empty() {
        return None;
    }

    let id: Matrix<T> = Matrix::id(A.rows);

    let bs = id.into_basis();

    let mut list: Vec<Vector<T>> = Vec::new();

    for i in 0..bs.len() {
        let b = &bs[i];
        let b_inv = A.solve(b, &lu);
        list.push(b_inv.unwrap());
    }

    let A_inv = Matrix::from_basis(list);

    Some(
        A_inv
    )
}
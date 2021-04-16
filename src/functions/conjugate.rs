#![allow(dead_code, warnings)]
use crate::{Number, core::matrix::Matrix};



pub fn conjugate<T: Number>(A: &Matrix<T>, S: &Matrix<T>) -> Option<Matrix<T>> {

    let lu = S.lu();
    
    let S_inv = S.inv(&lu);
    
    if S_inv.is_none() {
       return None;
    }

    let inv = S_inv.unwrap();
    
    let K: Matrix<T> = &(&inv * A) * S;

    Some(K)
}
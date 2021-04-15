use std::mem::size_of;
use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{matrix, Number};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



pub fn transpose<T: Number>(A: &Matrix<T>) -> Matrix<T> {

    let mut t = Matrix::new(A.columns, A.rows);

    for i in 0..A.rows {
        for j in 0..A.columns {
            t[[j,i]] = A[[i,j]];
        }
    }

    t
}
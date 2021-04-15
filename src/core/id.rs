use std::mem::size_of;
use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{matrix, Number};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



pub fn id<T: Number>(size: usize) -> Matrix<T> {

    let zero = T::from_i32(0).unwrap();
    
    let mut data: Vec<T> = vec![zero; size * size];
    
    for i in 0..size {
        data[(size * i) + i] = T::from_i32(1).unwrap();
    }
    
    Matrix {
        data,
        rows: size,
        columns: size
    }
}

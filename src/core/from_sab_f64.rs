use std::mem::size_of;
use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{matrix, Number};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



pub fn from_sab_f64(rows: usize, columns: usize, data: &SharedArrayBuffer) -> Matrix<f64> {

    let mut m = Matrix::new(rows, columns);

    let d = Float64Array::new(data);
    
    let size = rows * columns;

    let mut v = vec![0.; size];

    for i in 0..size {
        v[i] = d.get_index(i as u32);
    }

    m.data = v;
    
    m
}
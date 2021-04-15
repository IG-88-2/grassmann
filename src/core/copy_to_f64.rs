use std::mem::size_of;
use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{matrix, Number};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



pub fn copy_to_f64(m: &Matrix<f64>, dst: &mut Float64Array) {

    for i in 0..m.rows {
        for j in 0..m.columns {
            let idx = i * m.columns + j;
            dst.set_index(idx as u32, m[[i,j]]);
        }
    }
}
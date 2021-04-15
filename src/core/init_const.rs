use std::mem::size_of;
use super::{lu::{ lu }, matrix::{Matrix, add, P_compact, Partition}, solve::solve_lower_triangular, vector::Vector};
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{matrix, Number};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



pub fn init_const<T: Number>(A: &mut Matrix<T>, c: T) {

    for i in 0..A.columns {

        for j in 0..A.rows {

            A[[j,i]] = c;
        }
    }
}

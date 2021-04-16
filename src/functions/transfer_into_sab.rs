extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use std::mem::size_of;
use crate::core::matrix::Matrix;



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn transfer_into_sab(A:&Matrix<f64>, B:&Matrix<f64>) -> SharedArrayBuffer {
    let sa = A.mem_size();
    let sb = B.mem_size();
    let sc = A.rows * B.columns * size_of::<f32>();
    let size = sa + sb + sc;

    unsafe {
        log(&format!("\n ready to allocated {} bytes in sab \n", size));
    }

    let mut s = SharedArrayBuffer::new(size as u32);
    
    let v = Float64Array::new(&s);
    v.fill(0., 0, v.length());

    let mut v1 = Float64Array::new_with_byte_offset(&s, 0);
    let mut v2 = Float64Array::new_with_byte_offset(&s, sa as u32); //should it be + 1 ?

    Matrix::<f64>::copy_to_f64(&A, &mut v1);
    Matrix::<f64>::copy_to_f64(&B, &mut v2);

    unsafe {
        log(&format!("\n allocated {} bytes in sab \n", size));
    }

    s
}
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array, Int32Array};
use std::{any::TypeId, mem::size_of};
use crate::{Number, core::matrix::Matrix};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

//trait into sab ?

pub fn transfer_into_sab<T:Number>(A:&Matrix<T>, B:&Matrix<T>) -> SharedArrayBuffer {

    let sa = A.mem_size();
    let sb = B.mem_size();
    let sa1 = A.rows * A.columns * size_of::<T>();
    let sb1 = B.rows * B.columns * size_of::<T>();
    let sc = A.rows * B.columns * size_of::<T>();

    unsafe {
        log(&format!("\n cmp 1 {} - {} \n", sa, sa1));
        log(&format!("\n cmp 2 {} - {} \n", sb, sb1));
        log(&format!("\n cmp 3 {} \n", sc));
    }

    let size = sa + sb + sc;

    unsafe {
        log(&format!("\n ready to allocated {} bytes in sab \n", size));
    }

    let mut s = SharedArrayBuffer::new(size as u32);
    let id = TypeId::of::<T>();
    
    if id == TypeId::of::<f32>() {

        unsafe {
            log(&format!("\n convert f32 \n"));
        }

        let v = Float32Array::new(&s);

        v.fill(0., 0, v.length());
    
        let mut v1 = Float32Array::new_with_byte_offset(&s, 0);
        let mut v2 = Float32Array::new_with_byte_offset(&s, sa as u32); //should it be + 1 ?
    
        let A: Matrix<f32> = A.cast();
        let B: Matrix<f32> = B.cast();

        Matrix::<f32>::copy_to_f32(&A, &mut v1);
        Matrix::<f32>::copy_to_f32(&B, &mut v2);
    
        unsafe {
            log(&format!("\n allocated {} bytes in sab \n", size));
        }
    
        return s;

    } else if id == TypeId::of::<f64>() {

        unsafe {
            log(&format!("\n convert f64 \n"));
        }

        let v = Float64Array::new(&s);

        v.fill(0., 0, v.length());
    
        let mut v1 = Float64Array::new_with_byte_offset(&s, 0);
        let mut v2 = Float64Array::new_with_byte_offset(&s, sa as u32);

        let A: Matrix<f64> = A.cast();
        let B: Matrix<f64> = B.cast();

        Matrix::<f64>::copy_to_f64(&A, &mut v1);
        Matrix::<f64>::copy_to_f64(&B, &mut v2);
    
        return s;

    //int    
    } else {

        unsafe {
            log(&format!("\n convert int 32 \n"));
        }
    
        let v = Int32Array::new(&s);

        v.fill(0, 0, v.length());
    
        let mut v1 = Int32Array::new_with_byte_offset(&s, 0);
        let mut v2 = Int32Array::new_with_byte_offset(&s, sa as u32);

        let A: Matrix<i32> = A.cast();
        let B: Matrix<i32> = B.cast();

        Matrix::<i32>::copy_to_i32(&A, &mut v1);
        Matrix::<i32>::copy_to_i32(&B, &mut v2);
    
        return s;

    }
    
}
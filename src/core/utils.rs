#![allow(dead_code, warnings)]
use std::{f32::EPSILON, mem::size_of, ops::Index, time::Instant};
use crate::Number;
use super::matrix::Matrix;
use js_sys::{Float64Array, SharedArrayBuffer};
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use wasm_bindgen::{JsCast, prelude::*};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn transfer_into_sab(A:&Matrix<f64>, B:&Matrix<f64>) -> SharedArrayBuffer {
    let sa = A.mem_size();
    let sb = B.mem_size();
    let sc = A.rows * B.columns * size_of::<f64>();
    let size = sa + sb + sc;

    unsafe {
        log(&format!("\n ready to allocated {} bytes in sab \n", size));
    }
    let mut s = SharedArrayBuffer::new(size as u32);
    let mut v1 = Float64Array::new_with_byte_offset(&s, 0);
    let mut v2 = Float64Array::new_with_byte_offset(&s, sa as u32);
    
    Matrix::<f64>::copy_to_f64(&A, &mut v1);
    Matrix::<f64>::copy_to_f64(&B, &mut v2);

    unsafe {
        log(&format!("\n allocated {} bytes in sab \n", size));
    }

    s
}



pub fn pack_mul_task(t: [usize; 8], sab:&SharedArrayBuffer, A:&Matrix<f64>, B:&Matrix<f64>) -> js_sys::Array {
    let array: js_sys::Array = js_sys::Array::new();

    array.push(&sab);
    
    let a_rows = JsValue::from(A.rows as u32);
    let a_columns = JsValue::from(A.columns as u32);

    let b_rows = JsValue::from(B.rows as u32);
    let b_columns = JsValue::from(B.columns as u32);

    let t0 = JsValue::from(t[0] as u32);
    let t1 = JsValue::from(t[1] as u32);
    let t2 = JsValue::from(t[2] as u32); 
    let t3 = JsValue::from(t[3] as u32); 

    let t4 = JsValue::from(t[4] as u32); 
    let t5 = JsValue::from(t[5] as u32); 
    let t6 = JsValue::from(t[6] as u32); 
    let t7 = JsValue::from(t[7] as u32); 

    array.push(&a_rows);
    array.push(&a_columns);

    array.push(&b_rows);
    array.push(&b_columns);

    array.push(&t0);
    array.push(&t1);
    array.push(&t2);
    array.push(&t3);

    array.push(&t4);
    array.push(&t5);
    array.push(&t6);
    array.push(&t7);

    array
}



pub fn division_level<T: Number>(A: &Matrix<T>, optimal_block_size: usize, threads: usize) -> usize {
    
    let mut s = optimal_block_size;

    if (s as i32) == -1 {
        s = 10000;
    }

    let total_size = A.size();

    let mut blocks = 1;

    let x = total_size / optimal_block_size;

    if x < 1 {

        blocks = 1;

    } else {

        let c = if x > threads { threads } else { x };

        let n = (c as f64).log(4.).ceil();

        blocks = (4. as f64).powf(n) as usize;
    }

    blocks
}



pub fn from_data_square <T: Number>(d: &Vec<f64>, size: usize) -> Matrix<T> {

    let data: Vec<T> = d.iter().map(|x| { T::from_f64(*x).unwrap() }).collect();

    let mut m = Matrix::new(size,size);

    m.from_vec(data);

    m
}



pub fn eq_eps_f64(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON as f64
}



pub fn is_wasm() -> bool {
    cfg!(target_arch = "wasm32") && cfg!(target_os = "unknown")
}



pub fn measure_time(f: fn()) -> u128 {

    let before = Instant::now();

    f();
    
    let t = before.elapsed().as_millis();

    println!("\n \n No discarding... Elapsed time: {:.2?} \n \n", t);

    t
}



pub fn clamp(min: f64, max: f64) -> Box<dyn Fn(f64) -> f64> {
    
    let f = move |n: f64| {
        if n < min { 
            min 
        } else if n > max { 
            max 
        } else { 
            n 
        }
    };

    Box::new(f)
}

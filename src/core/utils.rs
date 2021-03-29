#![allow(dead_code, warnings)]
use std::{f32::EPSILON, mem::size_of, ops::Index, time::Instant};
use crate::Number;
use super::matrix::Matrix;
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint8Array};
use num_traits::identities;
//use rand::prelude::*;
//use rand::Rng;
use wasm_bindgen::{JsCast, prelude::*};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
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



pub fn round(n:f64) -> f64 {
    let c = (2. as f64).powf(32.);
    (n * c).round() / c
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



pub fn eq_eps_f64(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON as f64
}

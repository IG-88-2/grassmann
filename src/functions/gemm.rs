extern crate wasm_bindgen;
extern crate num_cpus;
use std::{cell::RefCell, future::Future, mem::size_of, pin::Pin, rc::Rc, task::{Context, Poll}};

use num::ToPrimitive;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{Array, Atomics, Float32Array, Float64Array, Int32Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use rand::prelude::*;
use rand::Rng;
use web_sys::Event;

use crate::{Number, core::matrix::{Matrix, add}, workers::{WorkerOperation, Workers}};



//what i can conclude from matrices shapes that is going to help select correct mult method to boost perf ?

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn pack_mul_task(
    t: [usize; 8], 
    sab:&SharedArrayBuffer, 
    A_rows: u32,
    A_columns: u32,
    B_rows: u32,
    B_columns: u32
) -> js_sys::Array {
    let array: js_sys::Array = js_sys::Array::new();

    array.push(&sab);
    
    let a_rows = JsValue::from(A_rows);
    let a_columns = JsValue::from(A_columns);

    let b_rows = JsValue::from(B_rows);
    let b_columns = JsValue::from(B_columns);

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



pub fn get_optimal_depth<T: Number> (A: &Matrix<T>, optimal_element_size: usize) -> usize {

    assert_eq!(A.rows, A.columns, "A should be square matrix");

    let p = (optimal_element_size as f64).log(4.).ceil();

    let optimal_element_size = (4. as f64).powf(p) as usize;

    let size = A.rows * A.columns;

    if size < 64 {
        return 0;
    }
    
    let chunks = size / optimal_element_size;

    if chunks < 2 {
        return 4;
    }
    
    (chunks as f64).log(4.).ceil() as usize
}



//what exactly about matrix indicates that this is going to give advantage ?
pub fn mul_blocks<T: Number>(
    a: &mut Matrix<T>, 
    b: &mut Matrix<T>, 
    optimal_block_size: usize,
    discard_zero_blocks: bool,
    threads: usize
) -> Matrix<T> {

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<T> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<T> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    unsafe {
        log(&format!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks));
    }
    
    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;

    let mut acc: Matrix<T> = Matrix::new(A.rows, B.columns);

    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {
            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = k + y;
                let ay0 = i;
                let ay1 = i + y;
                
                let bx0 = j;
                let bx1 = j + y;
                let by0 = k;
                let by1 = k + y;
                
                if discard_zero_blocks {
                    let mut sa = T::zero();
                    let mut sb = T::zero();

                    for m in ay0..ay1 {
                        for n in ax0..ax1 {
                            sa += A[[m,n]];
                        }
                    }

                    if sa == T::zero() {
                        continue;
                    }
                    
                    for m in by0..by1 {
                        for n in bx0..bx1 {
                            sb += B[[m,n]];
                        }
                    }

                    if sb == T::zero() {
                        continue;
                    }
                }
                
                for m in ay0..ay1 {
                    for n in bx0..bx1 {
                        for p in ax0..ax1 {
                            acc[[m,n]] += A[[m,p]] * B[[p,n]];    
                        }
                    }
                }
            }
        }
    }

    acc
}



pub fn prepare_multiply_threads(
    a: &Matrix<i32>, 
    b: &Matrix<i32>, 
    optimal_block_size: usize,
    threads: usize
) -> (
    usize,
    Matrix<i32>,
    Matrix<i32>,
    Vec<[usize; 8]>
) {
    
    let mut tasks: Vec<[usize; 8]> = Vec::new();

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<i32> = Matrix::new(s,s); 

    A = &A + &a;

    let mut B: Matrix<i32> = Matrix::new(s,s); 

    B = &B + &b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;
    
    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {
            for k in (0..A.columns).step_by(y) {

                let ax0 = k;
                let ax1 = k + y;
                let ay0 = i;
                let ay1 = i + y;
                
                let bx0 = j;
                let bx1 = j + y;
                let by0 = k;
                let by1 = k + y;
                
                let t: [usize; 8] = [

                    ax0, ay0, ax1, ay1, bx0, by0, bx1, by1

                ];

                tasks.push(t);
            }
        }
    }

    (
        blocks,
        A,
        B,
        tasks
    )
}



pub fn multiply_threads(
    hc: usize,
    optimal_block_size: usize, 
    A: &Matrix<i32>, 
    B: &Matrix<i32>
) -> WorkerOperation<Matrix<i32>> {
    
    let (blocks, mut A, mut B, tasks) = prepare_multiply_threads(
        A,
        B,
        optimal_block_size,
        hc
    );
    
    let ar = A.rows;
    let ac = A.columns;
    let bc = B.columns;
    let sa = A.mem_size();
    let sb = B.mem_size();

    let sab_rc: Rc<SharedArrayBuffer> = Rc::new(
        Matrix::<i32>::transfer_into_sab(&A, &B)
    );

    let workers = Workers::new("./worker.js", tasks.len());
    let workers = Rc::new( RefCell::new(workers) );
    let mut list = workers.borrow_mut();
    
    list.work = tasks.len() as i32;

    for i in 0..tasks.len() {
        let task = tasks[i];
        let worker = &mut list.ws[i];
        let sab = sab_rc.clone();
        let array = pack_mul_task(task, &sab, A.rows as u32, A.columns as u32, B.rows as u32, B.columns as u32);
        let hook = workers.clone();

        let c = Box::new(
            move |event: Event| {

                let mut list = hook.borrow_mut();

                list.ws[i].cb = None;

                list.work -= 1;
                
                if list.work == 0 {
                    let result = Int32Array::new_with_byte_offset(&sab, (sa + sb) as u32);
                    let data = result.to_vec();
                    let mut ma: Matrix<i32> = Matrix::new(ar, bc);
                    ma.set_vec(data);
                    
                    unsafe {
                        log(&format!("\n \n result {} \n \n", ma));
                    }
                    list.result = Some(ma);
                    list.waker.take().unwrap().wake();
                }
            }
        ) as Box<dyn FnMut(Event)>;
        
        let callback = Closure::wrap(c);
        
        worker.w.set_onmessage(
            Some(
                callback.as_ref().dyn_ref().unwrap()
            )
        );
        
        worker.cb = Some(callback);

        let result = worker.w.post_message(&array);
    }
    
    WorkerOperation {
        _ref: workers.clone(),
        extract: Box::new(
            move |s:&mut Workers<Matrix<i32>>| -> Option<Matrix<i32>> {
                let m = s.result.take();
                m
            }
        )
    }
}



#[wasm_bindgen]
pub fn ml_thread(
    sab:&SharedArrayBuffer,

    a_rows:usize,
    a_columns:usize,
    b_rows:usize,
    b_columns:usize,
    
    ax0:usize, ay0:usize,
    ax1:usize, ay1:usize,
    bx0:usize, by0:usize,
    bx1:usize, by1:usize
) {
    
    console_error_panic_hook::set_once();
    
    let s = size_of::<i32>();
    let sa = a_rows * a_columns * s;
    let sb = b_rows * b_columns * s;
    
    let ra = Int32Array::new_with_byte_offset(&sab, 0);
    let rb = Int32Array::new_with_byte_offset(&sab, sa as u32);
    let rc = Int32Array::new_with_byte_offset(&sab, (sa + sb) as u32);

    let vec1 = ra.to_vec();
    let vec2 = rb.to_vec();
    let vec3 = rc.to_vec();
    
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                    let v1: i32 = ra.get_index((m * a_columns + p) as u32);
                    let v2: i32 = rb.get_index((p * b_columns + n) as u32);
                    let result = v1.saturating_mul(v2);
                    unsafe{
                        Atomics::add(&rc, (m * b_columns + n) as u32, result);
                    }
                    //compare performance
                    //rc.set_index((m * b_columns + n) as u32, result);
                    //let old = Atomics::load(&rc2, (m * b_columns + n) as u32).unwrap(); 
                    //Atomics::store(&rc2, (m * b_columns + n) as u32, kkk as i32);
            }
        }
    }
}

//TODO discarding zero blocks
//TODO different partitioning strategy
#[wasm_bindgen]
pub async fn test_multiplication(hc: f64) {

    unsafe {
        log(&format!("\n hardware concurrency is {} \n", hc));
    }

    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let performance = window.performance().unwrap();
    let c = 12;
    let max = (2. as f64).powf(3.);

    
    
    for i in 10..c {
        let optimal_block_size = 5 * i;
        let max_side = 1000; //4 * i;
        let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
        //let mut B: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let mut B: Matrix<f64> = Matrix::rand(A.columns, A.rows, max);
        
        let mut A = A.cast();
        let mut B = B.cast();


        unsafe {
            log(&format!("\n multiplying A({}, {}) B({},{}) \n", A.rows, A.columns, B.rows, B.columns));
        }

        let start = performance.now();
        //TODO profile, when this is advantageous ?
        let r: Matrix<i32> = multiply_threads(hc as usize, optimal_block_size, &A, &B).await;
        //mul_blocks(&mut A, &mut B, optimal_block_size, false, hc as usize); 
        
        let end = performance.now();
        
        unsafe {
            log(&format!("\n by blocks {} \n", end - start));
        }

        let start = performance.now();
        let r2: Matrix<i32> = &A * &B;
        let end = performance.now();

        unsafe {
            log(&format!("\n naive {} \n", end - start));
        }




        unsafe {
            log(&format!("\n by blocks {} \n", r));
            log(&format!("\n naive {} \n", r2));
        }


        let mut r3: Matrix<f64> = Matrix::new(A.rows, B.columns);
        /*
        let r: Matrix<f64> = add(&r, &r3, 0); //TODO ? dim B
        
        if !(r == r2) {
            let diff: Matrix<f64> = &r - &r2;
            let s = diff.sum(); 
            unsafe {
                log(&format!("\n not equal, sum {} \n | r ({},{}) | r2 ({},{}) | {} \n", s, r.rows, r.columns, r2.rows, r2.columns, diff));

                //log(&format!("\n in threads {} \n in local {} \n", r, r2));
            }
            break;
        }
        */

        unsafe {
            //log(&format!("\n without threads {} \n", end - start));
        }
        
        //assert!(r == r2, "they should be equal {} \n \n {}", r, r2);

        unsafe {
            //log(&format!("\n final result is {} \n \n {} \n", r, r2));
        }
    }
}



mod tests {
    
    use super::{ Matrix, mul_blocks };

    //#[test]
    fn multiply_test() {
                
        let discard_zero_blocks = true;
        let threads = 10;

        for h in (1..2) {
            let max_side = 156;
            let max = 20000.;
            let optimal_block_size = 5000;
            let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
            let mut B: Matrix<f64> = Matrix::rand(A.columns, A.rows, max); //_shape(max_side, max);
        
            let C1: Matrix<f64> = &A * &B;
            let C: Matrix<f64> = mul_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, threads);

            assert_eq!(C.sum(), C1.sum(), "sum should be equal");

            for i in 0..C1.rows {
                for j in 0..C1.columns {
                    assert_eq!(C1[[i,j]], C1[[i,j]], "all entries should be equal");
                }
            }
        } 
    }
}

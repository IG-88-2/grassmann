use std::{any::TypeId, cell::RefCell, f32::EPSILON, future::Future, mem::*, pin::Pin, rc::Rc, task::{Context, Poll}};
use std::{
    collections::HashMap, 
    fmt,
    fmt::{
        Display, 
        Formatter
    }, 
    ops::{
        Add, 
        AddAssign, 
        Index, 
        IndexMut,
        Sub,
        SubAssign,
        Mul,
        MulAssign,
        Div,
        DivAssign,
        Neg
    }
};
extern crate wasm_bindgen;
extern crate num_cpus;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{Atomics, Float32Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, workers::Workers};
use super::{matrix::Matrix, matrix3::Matrix3, matrix4::Matrix4, utils::{division_level, pack_mul_task, transfer_into_sab}};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
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
    a: &Matrix<f32>, 
    b: &Matrix<f32>, 
    optimal_block_size: usize,
    threads: usize
) -> (
    usize,
    Matrix<f32>,
    Matrix<f32>,
    Vec<[usize; 8]>
) {
    
    let mut tasks: Vec<[usize; 8]> = Vec::new();

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<f32> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<f32> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    unsafe {
        log(&format!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks));
    }

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

                /*         
                for m in ay0..ay1 {
                    for n in bx0..bx1 {
                        for p in ax0..ax1 {
                            acc[[m,n]] += A[[m,p]] * B[[p,n]];    
                        }
                    }
                }
                */
                
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



pub struct WorkerOperation {
    pub _ref: Rc<RefCell<Workers>>,
    pub extract: Box<dyn FnMut(&mut Workers) -> Option<Matrix<f32>>>
}



impl Future for WorkerOperation {

    type Output = Matrix<f32>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let s = self._ref.clone();
        let mut state = s.borrow_mut();
        let result: Option<Matrix<f32>> = (self.extract)(&mut*state);

        if result.is_some() {
            let result = result.unwrap();
            state.terminate();
            Poll::Ready(result)
        } else {
            let w = cx.waker().clone();
            state.waker = Some(w);
            Poll::Pending
        }
    }
}



pub fn multiply_threads(
    hc: usize,
    optimal_block_size: usize, 
    A: &Matrix<f32>, 
    B: &Matrix<f32>
) -> WorkerOperation {
    let (blocks, mut A, mut B, tasks) = prepare_multiply_threads(
        A,
        B,
        optimal_block_size,
        hc
    );

    unsafe {
        log(&format!("\n multiply_threads: blocks {}, tasks {} \n", blocks, tasks.len()));
    }

    let ar = A.rows;
    let ac = A.columns;
    let bc = B.columns;
    let sa = A.mem_size();
    let sb = B.mem_size();
    let sab_rc: Rc<SharedArrayBuffer> = Rc::new(
        transfer_into_sab(&A, &B)
    );
    let workers = Workers::new("./worker.js", tasks.len());
    let workers = Rc::new( RefCell::new(workers) );
    let mut list = workers.borrow_mut();
    
    list.work = tasks.len() as i32;

    for i in 0..tasks.len() {
        let task = tasks[i];
        let worker = &mut list.ws[i];
        let sab = sab_rc.clone();
        let array = pack_mul_task(task, &sab, &A, &B);
        let hook = workers.clone();
        let c = Box::new(
            move |event: Event| {
                let mut list = hook.borrow_mut();
                list.ws[i].cb = None;
                list.work -= 1;
                
                if list.work == 0 {
                    let result = Float32Array::new_with_byte_offset(&sab, (sa + sb) as u32); //is offset correct ?
                    let data = result.to_vec(); //can this affect accuracy ?
                    let mut ma = Matrix::new(ar, bc);
                    ma.from_vec(data);
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
            move |s:&mut Workers| -> Option<Matrix<f32>> {
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
    let s = size_of::<f32>();
    let sa = a_rows * a_columns * s;
    let sb = b_rows * b_columns * s;
    
    let ra = Float32Array::new_with_byte_offset(&sab, 0);
    let rb = Float32Array::new_with_byte_offset(&sab, sa as u32);
    let rc = Float32Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    let rc2 = Uint8Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    
    //are tasks overlapping ? how to detect ?

    //TODO enable zero checks
    /*
    let mut ca = 0.;
    let mut cb = 0.;
    
    for m in ay0..ay1 {
        for n in ax0..ax1 {
            ca += ra.get_index((m * a_columns + n) as u32);
        }
    }

    if ca == 0. {
        return;
    }
    
    for m in by0..by1 {
        for n in bx0..bx1 {
            cb += rb.get_index((m * b_columns + n) as u32);
        }
    }

    if cb == 0. {
        return;
    }
    */
    
    /*
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                acc[[m,n]] += A[[m,p]] * B[[p,n]];    
            }
        }
    }
    */

    //no two threads output regions should overlap
    //confirm tasks non overlapping
    //can i sum them after multiplication of pieces done ?
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                unsafe {
                    /*
                    let v = rc.get_index((m * b_columns + n) as u32) + 
                            ra.get_index((m * a_columns + p) as u32) * 
                            rb.get_index((p * b_columns + n) as u32);

                    rc.set_index((m * b_columns + n) as u32, v);
                    */
                    let v: f32 = ra.get_index((m * a_columns + p) as u32) * rb.get_index((p * b_columns + n) as u32);

                    let old = rc.get_index((m * b_columns + n) as u32); //here is the issue
                    
                    if old != 0. {
                        unsafe {
                            log(&format!("\n old {} \n", old));
                        }
    
                    }
                    
                    rc.set_index((m * b_columns + n) as u32, old + v);

                    //Atomics::load(&rc2);
                    //add
                    //load
                    //float 8 consecutive 
                    //Atomics::store(&rc, (m * b_columns + n) as u32, v);
                }
            }
        }
    }
}
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
use js_sys::{Float64Array, SharedArrayBuffer, Uint32Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, workers::Workers};
use super::{matrix::Matrix, matrix3::Matrix3, matrix4::Matrix4, utils::{division_level, from_data_square, pack_mul_task, transfer_into_sab}};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn prepare_multiply_threads(
    a: &Matrix<f64>, 
    b: &Matrix<f64>, 
    optimal_block_size: usize,
    threads: usize
) -> (
    usize,
    Matrix<f64>,
    Matrix<f64>,
    Vec<[usize; 8]>
) {
    
    let mut tasks: Vec<[usize; 8]> = Vec::new();

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<f64> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<f64> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    println!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks);

    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;
    
    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {
            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);
                
                let t: [usize; 8] = [
                    ax0,ay0,
                    ax1,ay1,
                    bx0,by0,
                    bx1,by1
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



#[wasm_bindgen]
pub fn ml_thread(
    sab:&SharedArrayBuffer,

    a_rows:usize,
    a_columns:usize,
    b_rows:usize,
    b_columns:usize,
    
    ax0:usize,ay0:usize,
    ax1:usize,ay1:usize,
    bx0:usize,by0:usize,
    bx1:usize,by1:usize
) {
    let s = size_of::<f64>();
    let mut sa = a_rows * a_columns * s;
    let mut sb = b_rows * b_columns * s;
    
    let mut ra = Float64Array::new_with_byte_offset(&sab, 0);
    let mut rb = Float64Array::new_with_byte_offset(&sab, sa as u32);
    let mut rc = Float64Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    
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
    
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                let v = rc.get_index((m * b_columns + n) as u32) + 
                        ra.get_index((m * a_columns + p) as u32) * 
                        rb.get_index((p * b_columns + n) as u32);

                rc.set_index((m * b_columns + n) as u32, v);
            }
        }
    }
}



pub struct WorkerOperation {
    pub _ref: Rc<RefCell<Workers>>,
    pub extract: Box<FnMut(&mut Workers) -> Option<Matrix<f64>>>
}



impl Future for WorkerOperation {

    type Output = Matrix<f64>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let s = self._ref.clone();
        let mut state = s.borrow_mut();
        let result: Option<Matrix<f64>> = (self.extract)(&mut*state);

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

/*
TODO 
workers factory
workers bounded by hardware concurrency 
reuse workers
spread available work through workers, establish queue
*/

pub fn multiply_threads(
    hc: usize,
    optimal_block_size: usize, 
    A: &Matrix<f64>, 
    B: &Matrix<f64>
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
    let sa = A.mem_size();
    let sb = B.mem_size();
    let sab_rc: Rc<SharedArrayBuffer> = Rc::new(
        transfer_into_sab(&A, &B)
    );
    let mut workers = Workers::new("./worker.js", tasks.len());
    let mut workers = Rc::new( RefCell::new(workers) );
    let mut list = workers.borrow_mut();
    
    list.work = tasks.len() as u32;

    for i in 0..tasks.len() {
        let task = tasks[i];
        let worker = &mut list.ws[i];
        let sab = sab_rc.clone();
        let array = pack_mul_task(task, &sab, &A, &B);
        let hook = workers.clone();
        let c = Box::new(
            move |event: Event| {
                let sc = Rc::strong_count(&sab);
                let mut list = hook.borrow_mut();
                list.ws[i].cb = None;
                list.work -= 1;
                
                if list.work == 0 {
                    let mut result = Float64Array::new_with_byte_offset(&sab, (sa + sb) as u32);
                    let data = result.to_vec();
                    let mut ma = Matrix::new(ar, ac);
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
            move |s:&mut Workers| -> Option<Matrix<f64>> {
                let m = s.result.take();
                m
            }
        )
    }
}

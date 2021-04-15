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
use num::ToPrimitive;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{Atomics, Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, workers::Workers};
use super::{matrix::Matrix, matrix3::Matrix3, matrix4::Matrix4};

//what i can conclude from matrices shapes that is going to help select correct mult method to boost perf ?

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
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
    pub extract: Box<dyn FnMut(&mut Workers) -> Option<Matrix<f64>>>
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
    let bc = B.columns;
    let sa = A.mem_size();
    let sb = B.mem_size();
    let sab_rc: Rc<SharedArrayBuffer> = Rc::new(
        Matrix::<f64>::transfer_into_sab(&A, &B)
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
                    let result = Float64Array::new_with_byte_offset(&sab, (sa + sb) as u32); //is offset correct ?
                    let data = result.to_vec(); //can this affect accuracy ?
                    let mut ma = Matrix::new(ar, bc);
                    ma.set_vec(data);
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
    let rc2 = Uint32Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    
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

                    //let old = Atomics::load(&rc2, (m * b_columns + n) as u32).unwrap(); 
                    //let old = std::mem::transmute::<i32, f32>(old);
                    //let x = v + old;
                    //let x = std::mem::transmute::<f32, i32>(v);
                    let b = v.to_be_bytes(); //.to_i32().unwrap();
                    let kk = i32::from_be_bytes(b);
                    let r = Atomics::add(&rc2, (m * b_columns + n) as u32, kk);

                    /*
                    fn transmute<T, U>(e: T) -> U
                    std::mem::transmute::<&'b mut R<'static>, &'b mut R<'c>>(r)
                    */

                    //let u = v.to_bits();
                    //let y = f32::from_bits(old);
                    //let kk = y + v;
                    //let kkk = kk.to_bits();
                    
                    /*
                    ar buffer = new ArrayBuffer(4);
                    var intView = new Int32Array(buffer);
                    var floatView = new Float32Array(buffer);

                    floatView[0] = Math.PI
                    console.log(intView[0].toString(2)); //bits of the 32 bit float
                    */
                    //Atomics::store(&rc2, (m * b_columns + n) as u32, kkk as i32);
                    //rc.get_index((m * b_columns + n) as u32); //here is the issue
                    //let r = Atomics::add(&rc2, (m * b_columns + n) as u32, u as i32);

                    //r.unwrap();
                 
                    
                    /*
                    if old != 0 {
                        unsafe {
                            //log(&format!("\n old {} \n", old));
                        }
                    }
                    */
                    
                    //rc.set_index((m * b_columns + n) as u32, y + v);

                    //Atomics::store(&rc2, (m * b_columns + n) as u32, kkk as i32);
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
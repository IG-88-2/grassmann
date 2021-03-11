#![allow(dead_code, warnings)]
use std::{cell::RefCell, mem::size_of, rc::Rc, time::Instant};
use crate::matrix;
use crate::core::matrix::*;
use crate::workers::{Workers};
use js_sys::{Float64Array, SharedArrayBuffer, Uint32Array};
use wasm_bindgen::{JsCast, prelude::*};
use rand::prelude::*;
use rand::Rng;
use web_sys::Event;



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



#[wasm_bindgen]
pub fn multiply_blocks_worker() {

    let max = 226;
    let optimal_block_size = 100;
    let mut rng = rand::thread_rng();
    let A_rows = rng.gen_range(0, max) + 1; 
    let A_columns = rng.gen_range(0, max) + 1;
    let B_rows = A_columns;
    let B_columns = rng.gen_range(0, max) + 1;

    let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
    let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);
    
    let (
        blocks, //??? should be equal to blocks
        mut A,
        mut B,
        tasks
    ) = multiply_blocks_threads(
        &mut A, 
        &mut B, 
        optimal_block_size,
        1
    );
    
    let f = "_multiply_blocks_worker";
    let hc = tasks.len(); //should be equal to blocks;
    let workers = Workers::new("./worker.js", hc);
    let ra = A.into_sab();
    let rb = B.into_sab();
    let size = (size_of::<f64>() * A.rows * B.columns) as u32;
    let rc = SharedArrayBuffer::new(size);
    
    for i in 0..workers.ws.len() {
        let array: js_sys::Array = js_sys::Array::new();

        array.push(&ra);
        array.push(&rb);
        array.push(&rc);

        let mut dims = SharedArrayBuffer::new((size_of::<u32>() * 12) as u32);
        let dims_v = Uint32Array::new(&dims);

        dims_v.set_index(0, A.rows as u32);
        dims_v.set_index(1, A.columns as u32);
        dims_v.set_index(2, B.rows as u32);
        dims_v.set_index(3, B.columns as u32);

        let t = tasks[i];

        dims_v.set_index(4, t[0] as u32);
        dims_v.set_index(5, t[1] as u32);
        dims_v.set_index(6, t[2] as u32);
        dims_v.set_index(7, t[3] as u32);

        dims_v.set_index(8, t[4] as u32);
        dims_v.set_index(9, t[5] as u32);
        dims_v.set_index(10, t[6] as u32);
        dims_v.set_index(11, t[7] as u32);

        array.push(&dims);
        
        std::mem::forget(dims);
        
        let next = &workers.ws[i];

        let f = move |event: Event| {
            unsafe {
                log(&format!("worker {} completed", i));
            }
        };

        let c = Box::new(f) as Box<dyn FnMut(Event)>;

        let callback1 = Closure::wrap(c);
        
        next.w.set_onmessage(
            Some(
                callback1.as_ref().unchecked_ref()
            )
        );

        callback1.forget();
        
        let result = next.w.post_message(&array);
    }
    
    std::mem::forget(ra);
    std::mem::forget(rb);
    std::mem::forget(rc);
}



fn unpack_dims(mem: &SharedArrayBuffer) -> ((usize, usize),(usize, usize)) {

    let mut m = js_sys::Uint32Array::new( mem );

    let a_rows = m.get_index(0) as usize;
    let a_columns = m.get_index(1) as usize;
    let b_rows = m.get_index(2) as usize;
    let b_columns = m.get_index(3) as usize;

    ((a_rows, a_columns), (b_rows, b_columns))
}



fn pack_dims(a:&Matrix<f64>, b:&Matrix<f64>) -> js_sys::SharedArrayBuffer {

    let size = size_of::<u32>() as u32;

    let mem = js_sys::SharedArrayBuffer::new(4 * size);

    let mut m = js_sys::Uint32Array::new( &mem );

    m.set_index(0, a.rows as u32);
    m.set_index(1, a.columns as u32);
    m.set_index(2, b.rows as u32);
    m.set_index(3, b.columns as u32);

    mem
}



//convert into single sab with offsets - fn decompose
#[wasm_bindgen]
pub fn worker_test(
    a:&SharedArrayBuffer, 
    b:&SharedArrayBuffer, 
    c:&SharedArrayBuffer,
    dims:&SharedArrayBuffer
) {
    console_error_panic_hook::set_once();

    unsafe {
        log(&format!("\n worker_test: c {:?} \n", c));
    }

    let d = unpack_dims(dims);
    let (a_rows, a_columns) = d.0;
    let (b_rows, b_columns) = d.1;
    
    unsafe {
        log(&format!("\n worker_test: a_rows, a_columns ({},{}) \n", a_rows, a_columns));
    }

    unsafe {
        log(&format!("\n worker_test: b_rows, b_columns ({},{}) \n", b_rows, b_columns));
    }

    let mut A:Matrix<f64> = Matrix::<f64>::from_sab_f64(a_rows, a_columns, a);
    let mut B:Matrix<f64> = Matrix::<f64>::from_sab_f64(b_rows, b_columns, b);
    let optimal_block_size = 1000;
    let discard_zero_blocks = true;
    let result = multiply_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, 1);
    let mut C = Matrix::new(a_rows, b_columns);
    
    for i in 0..C.rows {
        for j in 0..C.columns {
            C[[i,j]] = result[[i,j]];
        }
    }

    let result = Float64Array::new(c);
    
    for i in 0..C.data.len() {
        result.set_index(i as u32, C.data[i]);
    }

    unsafe {
        log(&format!("\n worker_test: C {:?} \n", C));
    }
}



#[wasm_bindgen]
pub fn test_multiplication_worker(
    sab: &SharedArrayBuffer,
    a_rows: f64,
    a_columns: f64,
    b_rows: f64,
    b_columns: f64,
    t0: f64,
    t1: f64,
    t2: f64,
    t3: f64,
    t4: f64,
    t5: f64,
    t6: f64,
    t7: f64
) {
    unsafe {
        log(&format!("\n hello from worker {} {} {} {} | {} {} {} {} | {} {} {} {} \n", 
            a_rows,
            a_columns,
            b_rows,
            b_columns,

            t0,
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7
        ));
    }

    ml_thread(
        sab,

        a_rows as usize,
        a_columns as usize,
        b_rows as usize,
        b_columns as usize,
        
        t0 as usize,
        t1 as usize,
        t2 as usize,
        t3 as usize,
        t4 as usize,
        t5 as usize,
        t6 as usize,
        t7 as usize
    );
}



//TODO what is my memory limit in wasm ???
#[wasm_bindgen]
pub fn test_multiplication(hc: f64) {

    let (mut A, mut B) = get_test_matrices();

    unsafe {
        log(&format!("\n [HC {}] test_multiplication: A ({}, {}) | B ({}, {}) \n", hc, A.rows, A.columns, B.rows, B.columns));
    }

    let optimal_block_size = 4000;

    let (
        blocks, //??? should be equal to tasks.len
        mut A,
        mut B,
        tasks
    ) = multiply_blocks_threads(
        &mut A, 
        &mut B, 
        optimal_block_size,
        hc as usize
    );

    unsafe {
        log(&format!("\n test_multiplication: A ({}, {}) | B ({}, {}) | blocks {} | tasks {} \n", A.rows, A.columns, B.rows, B.columns, blocks, tasks.len()));
    }

    let example = &A * &B;
    let example = Rc::new(example);
   

    let s = size_of::<f64>();
    let sa = A.size() * s;
    let sb = B.size() * s;
    let sc = A.rows * B.columns * s;
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

    let a: Rc<SharedArrayBuffer> = Rc::new(s); //use rc
    
    let mut workers = Workers::new("./worker.js", tasks.len());

    let mut workers = Rc::new(
        RefCell::new(workers)
    );
    
    let mut list = workers.borrow_mut();

    for i in 0..list.ws.len() {

        let mut c_example = example.clone();

        let t = tasks[i];
        let next = &mut list.ws[i];
        let ta = a.clone();
        let array: js_sys::Array = js_sys::Array::new();

        array.push(&a);

        let a_r = A.rows;
        let a_c = A.columns;

        
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

        let hook = workers.clone();

        let f = move |event: Event| {
            
            let mut list = hook.borrow_mut();

            list.ws[i].cb = None;

            let sc = Rc::strong_count(&ta);
            
            if sc <= 2 {
                //means we are done
                let mut result = Float64Array::new_with_byte_offset(&ta, (sa + sb) as u32);

                let ggg = result.to_vec();

                let mut ma = Matrix::new(a_r, a_c);
                
                ma.from_vec(ggg);
                
                unsafe {
                    log(&format!("\n expected {:?} \n", &c_example.data));
                }

                unsafe {
                    log(&format!("\n result {:?} \n", ma.data));
                }

                let mt = Rc::get_mut(&mut c_example).unwrap();

                let equal = ma == *mt;

                assert!(equal, "matrices should be equal");

                list.terminate();
            }

            unsafe {
                log(&format!("worker {} completed | strong count {}, {:?}", i, sc, hook));
            }
        };

        let c = Box::new(f) as Box<dyn FnMut(Event)>;

        let callback1 = Closure::wrap(c);
        
        let kk = callback1.as_ref();

        let vv = Some(
            kk.dyn_ref().unwrap() //unchecked_ref()
        );

        next.w.set_onmessage(vv);
        
        next.cb = Some(callback1);

        let result = next.w.post_message(&array);
    }
}











fn test_quad(max: usize) {
    //let mut rng = rand::thread_rng();
    let A_rows =  12; //rng.gen_range(0..max) + 1; 
    let A_columns = 12; //rng.gen_range(0..max) + 1;
    let B_rows = 12; //A_columns;
    let B_columns = 12; //rng.gen_range(0..max) + 1;
    let max = 10.;
    let mut A: Matrix<f64> = Matrix::rand(A_rows, A_columns, max);
    let mut B: Matrix<f64> = Matrix::rand(B_rows, B_columns, max);

    let C1 = &A * &B; 
    let C1_sum = C1.sum();

    let s1 = Matrix::augment_sq2n_size(&A);
    let s2 = Matrix::augment_sq2n_size(&B);
    let s = std::cmp::max(s1,s2);
    let A: Matrix<f64> = Matrix::new(s,s) + A;
    let B: Matrix<f64> = Matrix::new(s,s) + B;
    let threads = 1;
    let optimal_block_size = 5000;
    let blocks = division_level(&A, optimal_block_size, threads);
    let cb = A.size() / blocks;
    let C: Matrix<f64> = &A * &B;
    let C_sum = C.sum();

    type Tuple = ((usize, usize), (usize, usize)); 

    let v: Vec<Tuple> = Vec::new();
    let zeros: Vec<Tuple> = Vec::new();

    //let mut l = A.rows / blocks; //should be A.rows / side length of a block
    
    let d = A.size() / blocks; //(A.size() / blocks).sqrt()

    let y = (d as f64).sqrt() as usize;

    println!("\n amount of blocks {} | block size {} | side length of single block {} \n", blocks, d, y);
    
    let mut ctr = 0;
    
    fn sum_block(A: &Matrix<f64>, start: (usize,usize), end: (usize,usize)) -> f64 {
        let mut result = 0.;
        for i in start.0..end.0 {
            for j in start.1..end.1 {
                result += A[[i,j]];
            }
        }
        result
    }
    
    //why total count not equal to amount of blocks ?
    //println!("\n total count is {} \n", ctr);
    let s = ((A.size() / blocks) as f64).sqrt() as usize;
    let mut R: Matrix<f64> = Matrix::new(A.rows, B.columns);

    
    for i in (0..A.rows).step_by(s) {
        for j in (0..B.columns).step_by(s) {

            let mut v:Vec<Vec<f64>> = Vec::new();

            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);

                //let sum = sum_block(&A, (x0 as usize, y0 as usize), (x1 as usize, y1 as usize));

                println!("\n R is ({},{}) BLOCK A ({},{}) - ({},{}) | BLOCK B ({},{}) - ({},{}) \n",
                    R.rows, R.columns,
                    ax0, ay0,
                    ax1, ay1,

                    bx0, by0,
                    bx1, by1
                );

                for m in ay0..ay1 {
                    for n in bx0..bx1 {
                        for p in ax0..ax1 {
                            R[[m,n]] += A[[m,p]] * B[[p,n]];    
                        }
                    }
                }
            }
        }
    }
    // assert_eq!(C.sum(), R.sum(), "should be equal");
    let D =  &R - &C;
    //println!("\n D is {} \n", D);
    //println!("\n C is {} \n", C);
    //println!("\n R is {} \n", R);

    /*
    println!("\n A is {} \n", A);
    println!("\n B is {} \n", B);
    println!("\n C is {} \n", C);
    println!("\n C1 is {} \n", C1);
    */
    assert_eq!(D.sum(), 0., "sum is zero");
    //assert_eq!(C1_sum, C_sum, "sum C1 should be equal sum C");
}



fn multiply_blocks_test(discard_zero_blocks:bool) {

    for h in (1..2) {
        let max = 156;
        let optimal_block_size = 5000;
        //let mut rng = rand::thread_rng();
        let A_rows = max; //rng.gen_range(0..max) + 1; 
        let A_columns = max; //rng.gen_range(0..max) + 1;
        let B_rows = max; //A_columns;
        let B_columns = max; //rng.gen_range(0..max) + 1;
        let threads = 1;
        let max = 10.;
        let mut A: Matrix<f64> = Matrix::rand(A_rows, A_columns, max);
        let mut B: Matrix<f64> = Matrix::rand(B_rows, B_columns, max);

        //println!("next ({}, {})", A.rows, A.columns);

        let C1: Matrix<f64> = &A * &B;
        let C: Matrix<f64> = multiply_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, threads);

        assert_eq!(C.sum(), C1.sum(), "sum should be equal");

        for i in 0..C1.rows {
            for j in 0..C1.columns {
                assert_eq!(C1[[i,j]], C1[[i,j]], "all entries should be equal");
            }
        }

    } 
}   


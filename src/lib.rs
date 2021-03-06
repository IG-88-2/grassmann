#![allow(warnings)]
mod core;
mod utils;
mod tests;
use crate::core::*;
extern crate wasm_bindgen;
use std::{any::TypeId, cell::RefCell, mem::size_of, ops::{Index, IndexMut}, pin::Pin, rc::Rc, sync::Arc};
use matrix::{transpose, multiply, Matrix, augment_sq2n};
use rand::prelude::*;
use rand::Rng;
use tests::get_test_matrices;
use utils::workers::{WorkerState, Workers};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{Array, Date, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use matrix::*;
use web_sys::{Document, Element, Event, HtmlElement, Navigator, WebGl2RenderingContext, WebGlProgram, WebGlShader, Window, Worker, window};
use web_sys::{
    ErrorEvent, 
    MessageEvent, 
    WebSocket
};
pub type Float = f64;



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
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



#[wasm_bindgen]
pub fn test_decomposition() {

    let t: Matrix<i32> = matrix![i32,
        3,3,1,2;
        3,3,1,2;
        4,4,5,2;
        4,4,5,2;
    ];

    let (A11, A12, A21, A22) = Matrix::decompose_blocks(t);
    
    unsafe {
        log(&format!("\n result \n {:?}, \n {:?}, \n {:?}, \n {:?}, \n", A11, A12, A21, A22));
    }
}



//TODO what is my memory limit in wasm ???
#[wasm_bindgen]
pub fn test_multiplication(hc: f64) {

    let (mut A,mut B) = get_test_matrices();

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

    let mut workers = Rc::new(RefCell::new(workers));
    
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
            
           
            if sc<=2 {
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

                let equal = eq(&ma, mt);

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





































//TODO async
//TODO arc
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
pub fn test5() {
    
    console_error_panic_hook::set_once();

    let worker = Worker::new("./worker.js").unwrap();

    let g = Array::new();

    let mut A: Matrix<f64> = Matrix::new(5,5);
    let mut B: Matrix<f64> = Matrix::new(5,5);
    let mut C: Matrix<f64> = Matrix::new(A.rows,B.columns); //unnecessary
    let dims = pack_dims(&A, &B);
    
    init_rand(&mut A);
    init_rand(&mut B);

    let C_test: Matrix<f64> = &A * &B;
    
    unsafe {
        log(&format!("\n expected C {:?} sum {}\n", C_test, C_test.sum()));
    }

    init_const(&mut C);

    unsafe {
        log(&format!("\n start with C {:?} \n", C));
    }

    let mut A_sab = A.into_sab();
    let mut B_sab = B.into_sab();
    let mut C_sab = C.into_sab();

    g.push(&A_sab);
    g.push(&B_sab);
    g.push(&C_sab);
    g.push(&dims);
    
    let p = &C_sab as *const SharedArrayBuffer;
    let C_ptr = &C_test as *const Matrix<f64>;
    
    let callback1 = Closure::wrap(
    Box::new(
            move |event: Event| {
                unsafe {
                    let original = &*C_ptr;
                    let mut m: Matrix<f64> = Matrix::<f64>::from_sab_f64(A.rows, B.columns, &*p);

                    log(&format!("\n expected {:?} \n", original));

                    log(&format!("\n worker done, result in main thread {:?} sum {} \n", m, m.sum()));
                    
                    //assert!(original==&m, "they should be equal");
                }
            }
        ) as Box<dyn FnMut(Event)>
    );
    
    worker.set_onmessage(
        Some(
            callback1.as_ref().unchecked_ref()
        )
    );

    callback1.forget();
    
    std::mem::forget(A_sab);
    std::mem::forget(B_sab);
    std::mem::forget(C_sab);
    std::mem::forget(C_test);
    std::mem::forget(dims);


    let result = worker.post_message(g.as_ref());

    std::mem::forget(g);

}



#[wasm_bindgen]
pub fn test4() {

    //create shared array buffer (A,B,C) 
    //fill it with numbers
    //send to worker
    //in worker - copy data from shared worker to wasm memory
    //perform computation
    //write result in corresponding cells in C
    
    console_error_panic_hook::set_once();

    let x = SharedArrayBuffer::new(32);
    let p = &x as *const SharedArrayBuffer;
    let xv = Float64Array::new(&x);
    xv.set_index(0, 3.456);


    let worker = Worker::new("./worker.js").unwrap();
    let arr = Array::new();
    arr.push(&x);
    let result = worker.post_message(arr.as_ref());


    let f =  Box::new(
        move |event: Event| {
            unsafe {
                let xv2 = Float64Array::new(&*p);
                let vec = xv2.to_vec();
                log(&format!("worker completed, result in main thread {:?}", vec));

            }
        }
    ) as Box<dyn FnMut(Event)>;

    let callback1 = Closure::wrap(f);
    
    worker.set_onmessage(
        Some(
            callback1.as_ref().unchecked_ref()
        )
    );

    callback1.forget();






   

    std::mem::forget(x);
}








#[wasm_bindgen]
pub fn mult(n:&SharedArrayBuffer) -> f64 { //*mut js_sys::SharedArrayBuffer
    
    unsafe {
        //let x = *n;
        log(&format!("mult {:?}", n));

        let mut m = js_sys::Float64Array::new(n);
        
        let mut vec = m.to_vec();

        //log(&format!("m {:?}", m));
        log(&format!("vec in thread {:?}", vec));

        //vec[0] = 1.333;

        
    }
    12.
}



/*
const memory = new WebAssembly.Memory({
  initial: 80,
  maximum: 80,
  shared: true
});
*/

#[wasm_bindgen]
pub fn test3() {

    unsafe {
        log(&format!("start test3"));
    }
    console_error_panic_hook::set_once();
    let arr = Array::new();
    //arr.push(&wasm_bindgen::module());
    // TODO: memory allocation error handling here is hard:
    //
    // * How to we make sure that our strong ref made it to a client
    //   thread?
    // * Need to handle the `?` on `post_message` as well.
    //arr.push(&wasm_bindgen::memory());


    //firstly i have to verify that my shared array available in worker and correctly displays values
    let worker = Worker::new("./worker.js").unwrap();
    let x = SharedArrayBuffer::new(32);
    let p = &x as *const SharedArrayBuffer; 
    let xv = Float64Array::new(&x);
    xv.set_index(0, 3.456);
    drop(xv);
    
    let f =  Box::new(
        move |event: Event| {
            unsafe {
               
                //let k = i.to_vec();
                //let y = *rrr;
                //
                let y = p;
                let j = &*y;
                let xv2 = Float64Array::new(j);
                let vec = xv2.to_vec();
                log(&format!("Hello from worker 222! {:?}", vec)); //it is empty

                //conclusion - apparently after being transferred into worker buffer released here ? 

                /*log(&format!("Hello from worker 222! {:?}", d)); 
                let mut m = js_sys::Float64Array::new( &*d );
        
                let mut vec =  m.to_vec();
                
                log(&format!("Hello from worker 222! {:?}", vec));*/
                //let rs = Arc::make_mut(&mut koko2); //Arc::into_raw(koko2);
                //let yh = rs.as_ptr(); //.entries();
                //let it = (*yh).get(0).dyn_into::<Uint8Array>().unwrap(); //unwrap_throw().unchecked_ref::<Uint8Array>().to_vec();
                //let mut gg: Vec<u8> = vec![0; it.length() as usize];
                //let mut sl = gg.as_mut_slice();
                
                //it.copy_to( sl);

                //log(&format!("Hello from worker it! {:?}", gg));

                /*
                for i in it {
                    i.entries();
                    log(&format!("Hello from worker NEXT! {:?}", i));
                }
                */
                //let yu = rs.as_ref(); //.unwrap(); //entries();
                //console_log!("unhandled event: {}", event.type_());
            }
        }
    ) as Box<dyn FnMut(Event)>;

    let callback1 = Closure::wrap(f);
    
    worker.set_onmessage(
        Some(
            callback1.as_ref().unchecked_ref()
        )
    );

    callback1.forget();

    
    //std::mem::forget(xv);
    unsafe {
        log(&format!("done"));
    }
    let result = worker.post_message(arr.as_ref());
    std::mem::forget(arr);
    std::mem::forget(x);
}

#[wasm_bindgen]
pub fn test2() {
    //create sab
    //enclose it into pinned arc with ref cell
    //get pointer to location
    //pass pointer to web worker
    //pass pointer back to rust
    //try to create view on shared memory
    
    //TODO how share pointer between rust contexts directly
    //TODO how memory structures [page - worker] [js - rust] [worker - rust]

    //try pin ? box ?
    //try move in closure
    //try manual drop

    //should i box this ?
    //try std::mem::forget
    let worker = Worker::new("./worker.js").unwrap();
    let x = SharedArrayBuffer::new(32);
    let p = &x as *const SharedArrayBuffer; 
    let xv = Float64Array::new(&x);
    xv.set_index(0, 3.456);

    //drop(x);
    
    //try leak box
    //try adding closure here

    
    unsafe {
        let mut m = js_sys::Float64Array::new( &*p );
        let mut vec = m.to_vec();
        log(&format!("vec start {:?}", vec));
    }

    let v = JsValue::from(p as u32);

    let result = worker.post_message(&v);
    //std::mem::forget(x);
}



#[wasm_bindgen]
pub fn test1() {
    //create sab
    //enclose it into pinned arc with ref cell
    //get pointer to location
    //pass pointer to web worker
    //pass pointer back to rust
    //try to create view on shared memory
    
    //TODO how share pointer between rust contexts directly
    //TODO how memory structures [page - worker] [js - rust] [worker - rust]

    //try pin ? box ?
    //try move in closure
    //try manual drop

    //should i box this ?
    //try std::mem::forget
    let worker = Worker::new("./worker.js").unwrap();
    let x = SharedArrayBuffer::new(32);
    
    let xv = Float64Array::new(&x);
    xv.set_index(0, 3.456);
    

    unsafe {
        let v = xv.to_vec();
        log(&format!("vec start {:?}", v));
    }

    //drop(xv); //?

    let c = RefCell::new(x);
    let v = Arc::new(c); //pin ?
    let d = Arc::into_raw(v);
    //do i really need array ?
    //let array: js_sys::Array = js_sys::Array::new();
    let g2 = JsValue::from(d as u32); //when im casting pointer to u32 is any data lost ? is pointer still valid ?
    //array.push(&g2);
    let result = worker.post_message(&g2);
}



#[wasm_bindgen]
pub fn test0(n: f64) -> f64 {

    unsafe {
        log(&format!("start"));
    }

    let discard_zero_blocks = true;
    let max = 16;
    let optimal_block_size = 5000;
    let mut rng = rand::thread_rng();

    let A_rows = rng.gen_range(0,max) + 1; 
    let A_columns = rng.gen_range(0,max) + 1;
    let B_rows = A_columns;
    let B_columns = rng.gen_range(0,max) + 1;

    let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
    let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);
    
    init_rand(&mut A);
    init_rand(&mut B);

    let C1: Matrix<f64> = &A * &B;

    let y = TypeId::of::<f64>;

    unsafe {
        //log(&format!("\n type ids {:?} {:?} {:?} \n", y, y(), y()));
        //log(&format!("\n first multiplication \n {:?} \n", C1));
    }
    
    let worker = Worker::new("./worker.js").unwrap();
    let mut result: Matrix<f64> = Matrix::new(A.rows,B.columns);
    let array: js_sys::Array = js_sys::Array::new();

    /*let a = Matrix::<f64>::into_typed_array(&mut A);
    let b = Matrix::<f64>::into_typed_array(&mut B);
    let c = Matrix::<f64>::into_typed_array(&mut result);*/
    
    //each worker should receive pointer

    /*array.push(&a);
    array.push(&b);
    array.push(&c);*/

    let state = Arc::new(
        RefCell::new(array)
    );
    let d = state.clone();
    
    //shove arc inside timeout - is this going to make drop difference ?

    let data = d.as_ptr();


    

    unsafe {
        let r = data as u32; //data.as_ref().unwrap();
        
        let mut test_me: f64 = 56.;
        let test_me_ptr = &mut test_me as *mut f64;
        let g = JsValue::from(test_me_ptr as u32);
        let array2: js_sys::Array = js_sys::Array::new();

        let len = 16;
        let mut me = js_sys::SharedArrayBuffer::new(len as u32);
        let mut mem = Pin::new_unchecked(me);
        let mut m = js_sys::Float64Array::new( &mem );
        let mut m2 = js_sys::Float64Array::new( &mem );

        let d = &mut mem as *mut Pin<js_sys::SharedArrayBuffer>;



        let f = move |event: Event| {
            unsafe {
                log(&format!("Hello from worker 222!"));

                /*log(&format!("Hello from worker 222! {:?}", d));
                let mut m = js_sys::Float64Array::new( &*d );
        
                let mut vec =  m.to_vec();
                
                log(&format!("Hello from worker 222! {:?}", vec));*/
                //let rs = Arc::make_mut(&mut koko2); //Arc::into_raw(koko2);
                //let yh = rs.as_ptr(); //.entries();
                //let it = (*yh).get(0).dyn_into::<Uint8Array>().unwrap(); //unwrap_throw().unchecked_ref::<Uint8Array>().to_vec();
                //let mut gg: Vec<u8> = vec![0; it.length() as usize];
                //let mut sl = gg.as_mut_slice();
                
                //it.copy_to( sl);
    
                //log(&format!("Hello from worker it! {:?}", gg));
    
                /*
                for i in it {
                    i.entries();
                    log(&format!("Hello from worker NEXT! {:?}", i));
                }
                */
                //let yu = rs.as_ref(); //.unwrap(); //entries();
                //console_log!("unhandled event: {}", event.type_());
            }
        };

        let c = Box::new(f) as Box<dyn FnMut(Event)>;

        let callback1 = Closure::wrap(c);
        
        

        /*worker.set_onmessage(
            Some(
                callback1.as_ref().unchecked_ref()
            )
        );
        callback1.forget();*/


        let g2 = JsValue::from(d as u32);
        array2.push(&g2);
        array2.push(&m2);
        
        let result = worker.post_message(&array2);
        //worker.post_message_with_transfer(r, &Array::of1(&r)).unwrap(); 
        
        log(&format!("\n RESULT {:?} \n", result));
            
        //worker.post_message(r);
        //worker.post_message_with_transfer(r, r); //.unwrap_throw(); //&Array::of1(&array_buffer)
        
    }
    
    /*
    let C: Matrix<f64> = multiply_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks);

    assert_eq!(C.sum(), C1.sum(), "sum should be equal");

    for i in 0..C1.rows {
        for j in 0..C1.columns {
            assert_eq!(C1[[i,j]], C1[[i,j]], "all entries should be equal");
        }
    }
    */

    0.  
}

#![allow(warnings)]
use std::task::Waker;

use wasm_bindgen::prelude::Closure;
use web_sys::{
    Document,
    Element,
    Event,
    HtmlElement,
    Navigator,
    WebGl2RenderingContext,
    WebGlProgram,
    WebGlShader,
    Window,
    Worker,
    window
};
use web_sys::{
    ErrorEvent, 
    MessageEvent, 
    WebSocket
};

use crate::core::matrix::Matrix;

/*
queue
workers factory
workers tasks
workers bounded by hardware concurrency 
reuse workers
spread available work through workers, establish queue
*/

#[derive(Debug)]
pub struct WorkerState {
    pub w:Worker,
    pub cb:Option<Closure<dyn FnMut(Event)>>
}



#[derive(Debug)]
pub struct Workers {
    pub ws: Vec<WorkerState>,
    pub work: i32,
    pub result: Option<Matrix<f64>>,
    pub waker: Option<Waker>
}



impl Workers {

    pub fn new(path:&str, length: usize) -> Workers {

        let mut ws: Vec<WorkerState> = Vec::new();

        for i in 0..length {
            let next = Worker::new(path).unwrap();
            let d = WorkerState {
                w:next,
                cb:None
            };            
            ws.push(d);
        }
        
        Workers {
            ws,
            work: 0,
            result: None,
            waker: None
        }
    }



    pub fn terminate(&mut self) {
        
        for i in 0..self.ws.len() {
            self.ws[i].w.terminate();
        }

        self.ws.clear();
    }
}

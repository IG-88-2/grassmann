#![allow(warnings)]
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::Context;
use std::task::Poll;
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



#[derive(Debug)]
pub struct WorkerState {
    pub w:Worker,
    pub cb:Option<Closure<dyn FnMut(Event)>>
}



#[derive(Debug)]
pub struct Workers<T> {
    pub ws: Vec<WorkerState>,
    pub work: i32,
    pub result: Option<T>,
    pub waker: Option<Waker>
}



impl <T> Workers <T> {

    pub fn new(path:&str, length: usize) -> Workers<T> {

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



    pub fn reset(&mut self) {

        self.work = 0;
        self.result = None;
        self.waker = None;

        for i in 0..self.ws.len() {
            self.ws[i].cb = None;
        }

    }



    pub fn terminate(&mut self) {
        
        for i in 0..self.ws.len() {
            self.ws[i].w.terminate();
        }

        self.ws.clear();
    }

}



pub struct WorkerOperation<T> {
    pub _ref: Rc<RefCell<Workers<T>>>,
    pub extract: Box<dyn FnMut(&mut Workers<T>) -> Option<T>>
}



impl Future for WorkerOperation<Matrix<i32>> {

    type Output = Matrix<i32>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let s = self._ref.clone();
        let mut state = s.borrow_mut();
        let result: Option<Matrix<i32>> = (self.extract)(&mut*state);

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

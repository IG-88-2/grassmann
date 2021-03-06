use std::mem::size_of;
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::{Number, core::matrix::Matrix};



pub fn into_sab<T: Number>(A: &mut Matrix<T>) -> SharedArrayBuffer {

    let size = size_of::<T>();

    let len = A.size() * size;

    let mem = SharedArrayBuffer::new(len as u32);

    let mut m = Uint8Array::new( &mem );

    for(i,v) in A.data.iter().enumerate() {
        let next = v.to_ne_bytes();
        for j in 0..size {
            m.set_index((i * size + j) as u32, next[j]);
        }
    }

    mem
}

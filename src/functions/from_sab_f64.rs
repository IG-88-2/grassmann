use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use crate::core::matrix::Matrix;



pub fn from_sab_f64(rows: usize, columns: usize, data: &SharedArrayBuffer) -> Matrix<f64> {

    let mut m = Matrix::new(rows, columns);

    let d = Float64Array::new(data);
    
    let size = rows * columns;

    let mut v = vec![0.; size];

    for i in 0..size {
        v[i] = d.get_index(i as u32);
    }

    m.data = v;
    
    m
}
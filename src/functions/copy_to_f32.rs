use js_sys::{Float32Array, Float64Array};
use crate::core::matrix::Matrix;



pub fn copy_to_f32(m: &Matrix<f32>, dst: &mut Float32Array) {

    for i in 0..m.rows {
        for j in 0..m.columns {
            let idx = i * m.columns + j;
            dst.set_index(idx as u32, m[[i,j]]);
        }
    }
}
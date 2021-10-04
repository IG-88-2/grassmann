use js_sys::{Float32Array, Float64Array, Int32Array};
use crate::core::matrix::Matrix;



pub fn copy_to_i32(m: &Matrix<i32>, dst: &mut Int32Array) {

    for i in 0..m.rows {
        for j in 0..m.columns {
            let idx = i * m.columns + j;
            dst.set_index(idx as u32, m[[i,j]]);
        }
    }
}
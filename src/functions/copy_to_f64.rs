use js_sys::Float64Array;
use crate::core::matrix::Matrix;



pub fn copy_to_f64(m: &Matrix<f64>, dst: &mut Float64Array) {

    for i in 0..m.rows {
        for j in 0..m.columns {
            let idx = i * m.columns + j;
            dst.set_index(idx as u32, m[[i,j]]);
        }
    }
}
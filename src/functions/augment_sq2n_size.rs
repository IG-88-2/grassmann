use crate::{Number, core::matrix::Matrix, matrix};



pub fn augment_sq2n_size<T: Number>(A: &Matrix<T>) -> usize {

    if A.rows==A.columns && (A.rows as f32).log2().fract() == 0. {
       return A.rows;
    }

    let mut side = std::cmp::max(A.rows, A.columns);

    let l: f64 = (side as f64).log2();

    if l.fract() != 0. {
        side = (2. as f64).powf(l.ceil()) as usize;
    }
    
    side
}

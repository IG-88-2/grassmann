use crate::{Number, core::matrix::Matrix, matrix};



pub fn is_upper_hessenberg<T: Number>(A: &Matrix<T>) -> bool {
        
    if !A.is_square() {
       return false; 
    }

    for i in 0..(A.columns - 2) {
        for j in (i + 2)..A.rows {
            
            if T::to_f64(&A[[j, i]]).unwrap().abs() > f32::EPSILON as f64 {
               return false;
            }
        }
    }

    true
}
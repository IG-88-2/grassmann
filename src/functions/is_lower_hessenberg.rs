use crate::{Number, core::matrix::Matrix};



pub fn is_lower_hessenberg<T: Number>(A: &Matrix<T>) -> bool {

    let zero = T::from_f64(0.).unwrap();

    if !A.is_square() {
       return false; 
    }

    for i in (2..A.columns).rev() {
        for j in 0..(i - 1) {

            if T::to_f64(&A[[j, i]]).unwrap().abs() > f32::EPSILON as f64 {
               return false;
            }
        }
    }

    true
}

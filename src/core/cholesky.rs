use super::matrix::{Matrix, add, P_compact, Partition};
use crate::{matrix, Number};



pub fn cholesky<T: Number>(A: &Matrix<T>) -> Option<Matrix<T>> {

    let zero = T::from_f64(0.).unwrap();

    let mut L = Matrix::new(A.rows, A.columns);

    for i in 0..A.rows {

        for j in i..A.columns {

            let mut s = A[[i, j]];

            for k in 0..i {
                s -= L[[k, i]] * L[[k, j]]; 
            }
            
            if i == j {
                
                if s <= zero {
                    return None;
                }
                
                let r = T::to_f64(&s).unwrap().sqrt();

                L[[i, j]] = T::from_f64(r).unwrap();
                
            } else {
                
                L[[i, j]] = s / L[[i, i]];
            }
        }
    }

    Some(L)
}
use crate::{Number, core::matrix::Matrix, matrix};

use super::utils::eq_bound_eps;



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



mod tests {
    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };
    use super::{ eq_bound_eps };


    #[test]
    fn cholesky_test() {

        let test = 20;

        for i in 2..test {
            let size = i;

            let mut A: Matrix<f64> = Matrix::rand(size, size, 1.);
    
            let f = move |x: f64| { if x < 0. { -1. * x } else { x } };
    
            let mut id: Matrix<f64> = Matrix::id(size);
    
            id = id * 100.;
    
            A.apply(&f);
    
            A = A + id;
    
            let mut At: Matrix<f64> = A.transpose();
    
            let mut A: Matrix<f64> = &At * &A;
    
            let mut L = A.cholesky();
            
            //println!("\n A ({},{}) is {} \n", A.rows, A.columns, A);
            
            if L.is_none() {
    
                println!("no Cholesky for you!");
    
            } else {
    
                let L = L.unwrap();
                
                let Lt = L.transpose();
    
                let G = &Lt * &L;
    
                assert!(eq_bound_eps(&A, &G), "A and G should be equivalent");
            }
        }
    }
}

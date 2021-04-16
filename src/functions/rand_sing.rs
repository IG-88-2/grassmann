use std::cmp::min;
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use crate::{Number, core::matrix::Matrix};



pub fn rand_sing<T: Number>(size: usize, max:f64) -> Matrix<T> {
        
    let mut rng = rand::thread_rng();

    let mut A = Matrix::rand(size, size, max);
    
    if size <= 1 {
        return A;
    }

    let c: usize = rng.gen_range(0, (size - 1) as u32) as usize;

    let mut basis = A.into_basis();

    println!("\n c is {} \n", c);
    
    for i in 0..A.rows {
        basis[c][i] = T::from_f64(0.).unwrap();
    }

    for i in 0..A.columns {

        if i == c {
           continue;
        }
        
        let mut v = basis[i].clone();

        let s: f64 = rng.gen_range(0., 1.);

        v = v * T::from_f64(s).unwrap();

        for j in 0..A.rows {
            basis[c][j] = basis[c][j] + v[j];
        }
    }

    Matrix::from_basis(basis)
}



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };

    #[test]
    fn rand_sing_test() {
        
        let size = 5;

        let max = 5.;

        let A: Matrix<f64> = Matrix::rand_sing(size, max);

        assert!( A.rank() < 5, "A.rank() < 5");
    }
}
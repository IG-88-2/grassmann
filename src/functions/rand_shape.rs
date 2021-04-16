use std::cmp::min;
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use crate::{Number, core::matrix::Matrix};



pub fn rand_shape<T: Number>(max_side: usize, max:f64) -> Matrix<T> {
        
    let mut rng = rand::thread_rng();
    
    let rows = rng.gen_range(0, max_side) + 1; 

    let columns = rng.gen_range(0, max_side) + 1;

    Matrix::rand(rows, columns, max)
}
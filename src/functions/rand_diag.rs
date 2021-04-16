use std::cmp::min;
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use crate::{Number, core::{matrix::Matrix, vector::Vector}};



pub fn rand_diag<T: Number>(size: usize, max: f64) -> Matrix<T> {

    let mut A: Matrix<T> = Matrix::new(size,size);

    let v = Vector::rand(size as u32, max);

    A.set_diag(v);

    A
}

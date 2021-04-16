use rand::prelude::*;
use rand::Rng;
use crate::{Number, core::matrix::Matrix};



pub fn rand_perm<T: Number>(size: usize) -> Matrix<T> {

    let mut m: Matrix<T> = Matrix::id(size);

    let mut rng = rand::thread_rng();

    let n: u32 = rng.gen_range(1, (size + 1) as u32);

    for i in 0..n {
        let high = (size - 1) as u32;
        let j: u32 = rng.gen_range(0, high);
        m.exchange_rows(i as usize, j as usize);
    }

    m
}
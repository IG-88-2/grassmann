use num_traits::identities;
use crate::{Number, core::matrix::Matrix};



pub fn sum<T: Number>(A: &Matrix<T>) -> T {

    let mut acc = identities::zero();

    for i in 0..A.data.len() {
        acc = acc + A.data[i];
    }
    
    acc
}

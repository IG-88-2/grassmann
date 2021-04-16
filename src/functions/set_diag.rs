use std::cmp::min;
use crate::{Number, core::{matrix::Matrix, vector::Vector}};



pub fn set_diag<T: Number>(A: &mut Matrix<T>, v:Vector<T>) {

    let c = min(A.rows, A.columns);

    assert_eq!(v.data.len(), c, "set_diag - incorrect size");

    for i in 0..c {
        A[[i, i]] = v[i];
    }
}

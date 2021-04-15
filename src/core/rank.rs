use super::matrix::{Matrix, add, P_compact, Partition};
use crate::{matrix, Number};



pub fn rank<T: Number>(A: &Matrix<T>) -> u32 {

    if A.columns > A.rows {

        let At = A.transpose();

        let lu = At.lu();

        let rank = At.columns - lu.d.len();

        rank as u32

    } else {

        let lu = A.lu();

        let rank = A.columns - lu.d.len();

        rank as u32
    }
}
use crate::{Number, core::matrix::Matrix};



pub fn init_const<T: Number>(A: &mut Matrix<T>, c: T) {

    for i in 0..A.columns {

        for j in 0..A.rows {

            A[[j,i]] = c;
        }
    }
}

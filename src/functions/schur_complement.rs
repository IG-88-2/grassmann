use crate::{Number, Partition, core::matrix::Matrix};



pub fn schur_complement<T: Number>(p:&Partition<T>) -> Option<Matrix<T>> {

    let A11_lu = p.A11.lu();

    let A11_inv = p.A11.inv(&A11_lu);
    
    if A11_inv.is_none() {
        return None;
    }

    let A11_inv = A11_inv.unwrap();
    
    let result: Matrix<T> = &(&p.A21 * &A11_inv) * &p.A12;
    
    Some(result)
}
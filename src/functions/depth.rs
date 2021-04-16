use crate::{Number, core::matrix::Matrix};



pub fn depth<T: Number>(A: &Matrix<T>) -> i32 {
    
    assert!(A.rows==A.columns, "depth: matrix is not square");
    
    if A.rows < 4 {
       return 0;
    }

    let size = (A.rows * A.columns) as f64;

    let p = size.log(4.);

    assert!(p.fract()==0., "depth: matrix is not exact power of 4");
    
    p as i32
}

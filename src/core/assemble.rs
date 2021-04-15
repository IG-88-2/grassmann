use super::matrix::{Matrix, add, P_compact, Partition};
use crate::{matrix, Number};



pub fn assemble<T: Number>(p: &Partition<T>) -> Matrix<T> {

    //A11 r x r
    //A12 r x (n - r)
    //A21 (n - r) x r
    //A22 (n - r) x (n - r)

    let rows = p.A11.rows + p.A21.rows;
    
    let columns = p.A11.columns + p.A12.columns; 
    
    let mut A = Matrix::new(rows, columns);
    
    for i in 0..p.A11.rows {
        for j in 0..p.A11.columns {
            A[[i, j]] = p.A11[[i, j]];
        }
    }

    for i in 0..p.A12.rows {
        for j in 0..p.A12.columns {
            A[[i, j + p.A11.columns]] = p.A12[[i, j]];
        }
    }
    
    for i in 0..p.A21.rows {
        for j in 0..p.A21.columns {
            A[[i + p.A11.rows, j]] = p.A21[[i, j]];
        }
    }

    for i in 0..p.A22.rows {
        for j in 0..p.A22.columns {
            A[[i + p.A11.rows, j + p.A11.columns]] = p.A22[[i, j]];
        }
    }

    A
}
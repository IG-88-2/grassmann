use super::matrix::{Matrix, add, P_compact, Partition};
use crate::{matrix, Number};



pub fn partition<T: Number>(A: &Matrix<T>, r: usize) -> Option<Partition<T>> {
        
    if r >= A.columns || r >= A.rows {
        return None;
    }

    //A11 r x r
    //A12 r x (n - r)
    //A21 (n - r) x r
    //A22 (n - r) x (n - r)

    let mut A11: Matrix<T> = Matrix::new(r, r);
    let mut A12: Matrix<T> = Matrix::new(r, A.columns - r);
    let mut A21: Matrix<T> = Matrix::new(A.rows - r, r);
    let mut A22: Matrix<T> = Matrix::new(A.rows - r, A.columns - r);

    for i in 0..r {
        for j in 0..r {
            A11[[i,j]] = A[[i, j]];
        }
    }

    for i in 0..r {
        for j in 0..(A.columns - r) {
            A12[[i,j]] = A[[i,j + r]];
        }
    }

    for i in 0..(A.rows - r) {
        for j in 0..r {
            A21[[i,j]] = A[[i + r, j]];
        }
    }

    for i in 0..(A.rows - r) {
        for j in 0..(A.columns - r) {
            A22[[i,j]] = A[[i + r, j + r]];
        }
    }

    Some(
        Partition {
            A11,
            A12,
            A21,
            A22
        }
    )
}
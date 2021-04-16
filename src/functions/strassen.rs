use crate::{Number, core::matrix::Matrix};
use super::utils::eq_bound_eps;



pub fn decompose_blocks<T: Number>(A: &Matrix<T>) -> (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>) {
    
    assert_eq!(A.rows, A.columns, "matrix should be square");
    assert_eq!((A.rows as f32).log2().fract(), 0., "matrix should be pow 2");

    let r = A.rows / 2;
    let c = A.columns / 2;
    
    let mut A11 = Matrix::new(r, c);
    let mut A12 = Matrix::new(r, c);
    let mut A21 = Matrix::new(r, c);
    let mut A22 = Matrix::new(r, c);
    
    for i in 0..r {
        for j in 0..c {
            A11[[i, j]] = A[[i, j]];
            A12[[i, j]] = A[[i, c + j]];
            A21[[i, j]] = A[[r + i, j]];
            A22[[i, j]] = A[[r + i, c + j]];
        }
    }


    (A11, A12, A21, A22)
}



pub fn recombine_blocks<T: Number>(A11: Matrix<T>, A12: Matrix<T>, A21: Matrix<T>, A22: Matrix<T>) -> Matrix<T> {

    assert_eq!(A11.rows, A11.columns, "A11 matrix should be square");
    assert_eq!((A11.rows as f32).log2().fract(), 0., "A11 matrix should be pow 2");
    
    assert_eq!(A11.rows, A12.rows, "A11 should have rows equivalent to A12");
    assert_eq!(A11.columns, A12.columns, "A11 should have columns equivalent to A12");
    
    assert_eq!(A11.rows, A21.rows, "A11 should have rows equivalent to A21");
    assert_eq!(A11.columns, A21.columns, "A11 should have columns equivalent to A21");

    assert_eq!(A11.rows, A22.rows, "A11 should have rows equivalent to A22");
    assert_eq!(A11.columns, A22.columns, "A11 should have columns equivalent to A22");

    let rows = A11.rows;
    let columns = A11.columns;
    let r = rows * 2;
    let c = columns * 2;

    let mut A = Matrix::new(r, c);

    for i in 0..rows {
        for j in 0..columns {
            A[[i,j]] = A11[[i,j]];
            A[[i,j + columns]] = A12[[i,j]];
            A[[i + rows,j]] = A21[[i,j]];
            A[[i + rows,j + columns]] = A22[[i,j]];
        }
    } 

    A
}



pub fn strassen<T: Number>(A:&Matrix<T>, B:&Matrix<T>) -> Matrix<T> {
    
    let (
        A11, 
        A12, 
        A21, 
        A22
    ) = decompose_blocks(A);

    let (
        B11, 
        B12, 
        B21, 
        B22
    ) = decompose_blocks(B);
    
    let P1: Matrix<T> = &(&A11 + &A22) * &(&B11 + &B22); //TODO implement with consumption
    let P2: Matrix<T> = &(&A21 + &A22) * &B11;
    let P3: Matrix<T> = &A11 * &(&B12 - &B22);
    let P4: Matrix<T> = &A22 * &(&B21 - &B11);
    let P5: Matrix<T> = &(&A11 + &A12) * &B22;
    let P6: Matrix<T> = &(&A21 - &A11) * &(&B11 + &B12);
    let P7: Matrix<T> = &(&A12 - &A22) * &(&B21 + &B22);

    let C11: Matrix<T> = &(&(&P1 + &P4) - &P5) + &P7;
    let C12: Matrix<T> = &P3 + &P5;
    let C21: Matrix<T> = &P2 + &P4;
    let C22: Matrix<T> = &(&(&P1 + &P3) - &P2) + &P6;
    let C = recombine_blocks(C11, C12, C21, C22);
    
    C
}



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use super::{ strassen, eq_bound_eps };
    use crate::{ matrix::{ Matrix }, matrix, vector };
    


    #[test]
    fn strassen_test() {
        
        let max_side = 10;
        let max = 10.;
        let a: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let b: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let s1 = Matrix::augment_sq2n_size(&a);
        let s2 = Matrix::augment_sq2n_size(&b);
        let s = std::cmp::max(s1,s2);

        let mut A: Matrix<f64> = Matrix::new(s,s); 

        A = &A + &a;

        let mut B: Matrix<f64> = Matrix::new(s,s); 

        B = &B + &b;

        let expected: Matrix<f64> = &A * &B;

        let s = format!("\n [{}] expected \n {:?} \n", expected.sum(), expected);

        println!("{}", s);
        
        let C = strassen(&A, &B);
        
        let s = format!("\n [{}] result \n {:?} \n", C.sum(), C);
        
        println!("{}", s);

        let equal = eq_bound_eps(&expected, &C);

        assert!(equal, "should be equal");
    }
}
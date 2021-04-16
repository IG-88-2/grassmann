use crate::{Number, core::matrix::Matrix};



pub fn is_diagonally_dominant<T: Number>(A: &Matrix<T>) -> bool {

    if A.rows != A.columns {
       return false;
    }

    let zero = T::from_f64(0.).unwrap();

    for i in 0..A.rows {

        let mut acc = zero;

        let mut p = zero;

        for j in 0..A.columns {
        
            if i == j {
               p = A[[i, j]];
            } else {
               acc += A[[i, j]];
            }
        }

        if p.abs() < acc.abs() {
            return false;
        }
    }
    
    true
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn is_diagonally_dominant() {

        let mut A: Matrix<i32> = Matrix::id(10);
        let mut N: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_diagonally_dominant(), true, "is_diagonally_dominant - identity is diagonally dominant");
        
        A = A * 10;
        N = N * -2;
        
        let mut C: Matrix<i32> = Matrix::new(10, 10);

        C.init_const(1);

        A = A + N;
        A = A + C;
        
        assert_eq!(A.is_diagonally_dominant(), true, "is_diagonally_dominant - after transformation");

        A[[0, 1]] += 1;

        assert_eq!(A.is_diagonally_dominant(), false, "is_diagonally_dominant - no more");
    }

}

use crate::{Number, core::matrix::Matrix};



pub fn is_diag<T: Number>(A: &Matrix<T>) -> bool {
        
    let zero = T::from_f64(0.).unwrap();

    for i in 0..A.rows {
        for j in 0..A.columns {
            if i == j {
                continue;
            }
            if A[[i, j]] != zero {
                return false;
            } 
        }
    }

    true
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn is_diag() {

        let mut A: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_diag(), true, "is_diag - identity is diagonal");
        
        A[[0, 9]] = 1;

        assert_eq!(A.is_diag(), false, "is_diag - A[[0, 9]] = 1, A no longer diagonal");

        A[[0, 9]] = 0;
        A[[5, 6]] = 1;

        assert_eq!(A.is_diag(), false, "is_diag - A[[5, 6]] = 1, A no longer diagonal");

        A[[5, 6]] = 0;

        assert_eq!(A.is_diag(), true, "is_diag - A is diagonal");
    }
}
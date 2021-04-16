use crate::{Number, core::matrix::Matrix};



pub fn is_identity<T: Number>(A: &Matrix<T>) -> bool {

    let zero = T::from_f64(0.).unwrap();
    
    let one = T::from_f64(1.).unwrap();

    for i in 0..A.rows {
        for j in 0..A.columns {
            if i == j {
                if A[[i, j]] != one {
                    return false;
                }
            } else {
                if A[[i, j]] != zero {
                    return false;
                }
            }
        }
    }

    true
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn is_identity() {
        
        let mut A: Matrix<i32> = Matrix::id(1);
        
        assert_eq!(A.is_identity(), true, "is_identity - 1x1 id");

        A[[0, 0]] = 0;

        assert_eq!(A.is_identity(), false, "is_identity - 1x1 id");

        let mut A: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_identity(), true, "is_identity - 10x10 id");

        A[[1, 1]] = 0;

        assert_eq!(A.is_identity(), false, "is_identity - 10x10 not id");

        A[[1, 1]] = 1;
        A[[1, 0]] = 1;

        assert_eq!(A.is_identity(), false, "is_identity - 10x10 not id");
    }
}
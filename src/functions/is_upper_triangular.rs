use crate::{Number, core::matrix::Matrix};



pub fn is_upper_triangular<T: Number>(A: &Matrix<T>) -> bool {
        
    if !A.is_square() {
        return false;
    }

    let zero = T::from_f64(0.).unwrap();
    
    for i in 0..A.rows {

        for j in 0..i {

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
    fn is_upper_triangular() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - identity is upper triangular");
        
        id[[5, 6]] = 1.;

        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - id[[6, 5]] = 1., identity is still upper triangular");

        id[[0, 9]] = 1.;
        
        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - id[[9, 0]] = 1., identity is still upper triangular");

        id[[9, 0]] = 1.;

        assert_eq!(id.is_upper_triangular(), false, "is_upper_triangular - id[[0, 9]] = 1., identity is no longer upper triangular");
    }
}
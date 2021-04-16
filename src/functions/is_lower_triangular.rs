use crate::{Number, core::matrix::Matrix};



pub fn is_lower_triangular<T: Number>(A: &Matrix<T>) -> bool {

    if !A.is_square() {
        return false;
    }

    let zero = T::from_f64(0.).unwrap();
    
    for i in 0..A.rows {

        for j in (i + 1)..A.columns {

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
    fn is_lower_triangular() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - identity is lower triangular");

        id[[6, 5]] = 1.;

        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - id[[6, 5]] = 1., identity is still lower triangular");

        id[[9, 0]] = 1.;
        
        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - id[[9, 0]] = 1., identity is still lower triangular");

        id[[0, 9]] = 1.;

        assert_eq!(id.is_lower_triangular(), false, "is_lower_triangular - id[[0, 9]] = 1., identity is no longer lower triangular");
    }
}
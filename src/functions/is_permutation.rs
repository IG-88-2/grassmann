use crate::{Number, core::matrix::Matrix};



pub fn is_permutation<T: Number>(A: &Matrix<T>) -> bool {

    if A.rows != A.columns || A.rows <= 1 {
       return false; 
    }

    for i in 0..(A.rows - 1) {

        let start = i + 1;

        for j in start..A.columns {
            
            if A[[i,j]] != A[[j,i]] {

                return false;
                
            }
        }
    }

    true
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn is_permutation() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation");

        id.exchange_rows(1, 5);
        id.exchange_rows(2, 7);
        id.exchange_rows(0, 9);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation after transform");

        id.exchange_columns(0, 9);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation after col exchange");

        id[[5, 6]] = 0.1;

        assert_eq!(id.is_permutation(), false, "is_permutation - identity is no longer permutation after augmentation");
    }

}
use crate::{Number, core::matrix::Matrix};



pub fn transpose<T: Number>(A: &Matrix<T>) -> Matrix<T> {

    let mut t = Matrix::new(A.columns, A.rows);

    for i in 0..A.rows {
        for j in 0..A.columns {
            t[[j,i]] = A[[i,j]];
        }
    }

    t
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn transpose() {
        
        let t: Matrix<i32> = matrix![i32,
            1,2,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,4,5;
            1,2,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,4,5;
        ];
        let c = t.clone();
        let t2 = c.transpose();
        let r = &t * &t2;
        
        assert_eq!(r.is_symmetric(), true, "product of transposed matrices should be symmetric {}", r);

        //TODO transpose orthonormal basis, inverse
        
    }

}
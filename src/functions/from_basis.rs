use crate::{Number, core::{matrix::Matrix, vector::Vector}};



pub fn from_basis<T: Number>(b: Vec<Vector<T>>) -> Matrix<T> {
    
    if b.len() == 0 {
       return Matrix::new(0, 0);
    }

    let rows = b[0].data.len();

    let equal = b.iter().all(|v| v.data.len() == rows);
    
    assert!(equal, "\n from basis: vectors should have equal length \n");

    let columns = b.len();
    
    let mut m = Matrix::new(rows, columns);

    for i in 0..columns {
        let next = &b[i];
        for j in 0..rows {
            m[[j, i]] = next[j];
        }
    }

    m
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };

    #[test]
    fn from_basis() {

        let A = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let b = vec![
            Vector::new(vec![1., 2., 3.]),
            Vector::new(vec![2., 4., 5.]),
            Vector::new(vec![3., 7., 3.])
        ];

        let R = Matrix::from_basis(b);

        assert_eq!(A, R, "from_basis: matrices should be equal");
    }

}
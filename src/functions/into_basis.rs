use crate::{Number, core::matrix::Matrix, core::vector::Vector};



pub fn into_basis<T: Number>(A: &Matrix<T>) -> Vec<Vector<T>> {

    let zero = T::from_f64(0.).unwrap();
    let mut b: Vec<Vector<T>> = Vec::new();

    for i in 0..A.columns {
        let mut v = Vector::new(vec![zero; A.rows]);
        for j in 0..A.rows {
            v[j] = A[[j, i]];
        }
        b.push(v);
    }

    b
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn into_basis() {

        let A = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let b = A.into_basis();

        for i in 0..b.len() {
            let col_i = &b[i];
            for j in 0..A.rows {
                assert_eq!(col_i[j], A[[j, i]], "entries should be equal");
            }
        }
    }
}
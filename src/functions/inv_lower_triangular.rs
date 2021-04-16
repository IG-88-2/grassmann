use crate::{Number, core::matrix::Matrix, core::vector::Vector};
use super::{solve::solve_lower_triangular, utils::eq_bound_eps};



pub fn inv_lower_triangular<T: Number>(A: &Matrix<T>) -> Option<Matrix<T>> {

    assert!(A.is_lower_triangular(), "matrix should be lower triangular");

    if A.rows != A.columns {
        return None;
    }

    let id: Matrix<T> = Matrix::id(A.rows);

    let bs = id.into_basis();

    let mut list: Vec<Vector<T>> = Vec::new();

    for i in 0..bs.len() {
        let b = &bs[i];
        let b_inv = solve_lower_triangular(A, b);
        if b_inv.is_none() {
            return None;
        }
        list.push(b_inv.unwrap());
    }

    let A_inv = Matrix::from_basis(list);

    Some(
        A_inv
    )
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };
    use super::{ eq_bound_eps };

    #[test]
    fn inv_lower_triangular() {
        
        let size = 10;
        let A: Matrix<f64> = Matrix::rand(size, size, 100.);
        let lu = A.lu();
        let L = lu.P * lu.L; 
        let id = Matrix::id(size);

        println!("\n L is {} \n", L);

        let L_inv = L.inv_lower_triangular().unwrap();

        let p: Matrix<f64> = &L_inv * &L;
        
        let equal = eq_bound_eps(&id, &p);

        println!("\n L_inv {} \n ID is {} \n", L_inv, p);

        assert!(equal, "inv_lower_triangular");
    }
}
use crate::{Number, core::{matrix::Matrix, vector::Vector}};
use super::{solve::solve_upper_triangular, utils::eq_bound_eps};



pub fn inv_upper_triangular<T: Number>(A: &Matrix<T>) -> Option<Matrix<T>> {

    //assert!(self.is_upper_triangular(), "matrix should be upper triangular");

    if A.rows != A.columns {
        return None;
    }

    let id: Matrix<T> = Matrix::id(A.rows);

    let bs = id.into_basis();

    let mut list: Vec<Vector<T>> = Vec::new();

    for i in 0..bs.len() {
        let b = &bs[i];
        let b_inv = solve_upper_triangular(A, b);
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

    //solve upper triangular
    #[test]
    fn inv_upper_triangular() {
        let size = 10;
        let A: Matrix<f64> = Matrix::rand(size, size, 100.);
        let lu = A.lu();
        let U = lu.U; 
        let id = Matrix::id(size);

        println!("\n U is {} \n", U);

        let U_inv = U.inv_upper_triangular().unwrap();

        let p: Matrix<f64> = &U_inv * &U;
        
        let equal = eq_bound_eps(&id, &p);

        println!("\n U_inv {} \n ID is {} \n", U_inv, p);

        assert!(equal, "inv_upper_triangular");
    }
}
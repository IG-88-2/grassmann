use crate::{Number, core::matrix::Matrix};

use super::utils::eq_bound_eps;



pub fn inv_diag<T: Number>(A: &Matrix<T>) -> Matrix<T> {

    assert!(A.is_diag(), "inv_diag matrix should be diagonal");

    assert!(A.is_square(), "inv_diag matrix should be square");

    let one = T::from_f64(1.).unwrap();
    
    let mut A_inv: Matrix<T> = Matrix::new(A.rows, A.columns);

    for i in 0..A.rows {
        A_inv[[i, i]] = one / A[[i, i]];
    }

    A_inv
}



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };
    use super::{ eq_bound_eps };

    #[test]
    fn inv_diag() {

        let length: usize = 10;

        let v: Vector<f64> = Vector::rand(length as u32, 100.);

        let mut m: Matrix<f64> = Matrix::new(length, length);

        m.set_diag(v);

        let m_inv = m.inv_diag();

        let p: Matrix<f64> = &m_inv * &m;

        let id = Matrix::id(length);

        let equal = eq_bound_eps(&id, &p);

        assert!(equal, "inv_diag");
    }

}
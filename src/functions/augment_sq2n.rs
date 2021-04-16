use crate::{Number, core::matrix::Matrix, matrix};
use super::gemm::get_optimal_depth;



pub fn augment_sq2n<T: Number>(A: &Matrix<T>) -> Matrix<T> {
    
    let side = A.augment_sq2n_size();

    if side == A.rows && side == A.columns {
       return A.clone();
    }
    
    let mut m: Matrix<T> = Matrix::new(side, side);

    m = &m + A;

    m
}



mod tests {
    use crate::{ matrix::{ Matrix } };
    use super::{ get_optimal_depth };

    #[test]
    fn augment_sq2n() {
        let iterations = 2;
        let max_side = 183;
        let max = 33333.;

        for i in 0..iterations {
            let m: Matrix<f64> = Matrix::rand_shape(max_side, max);
            let aug = Matrix::augment_sq2n(&m);
            let d = get_optimal_depth(&aug, 100);
            let size = (aug.rows * aug.columns) as f64;
    
            println!("({},{}) | size - {} | level - {}", aug.rows, aug.columns, size, d);
    
            let l: f64 = size.log2();

            assert_eq!(aug.rows, aug.columns, "from ({} {}) to ({} {}) - should be square", m.rows, m.columns, aug.rows, aug.columns);
            assert_eq!(l.fract(), 0., "from ({} {}) to ({} {}) - should be power of 2", m.rows, m.columns, aug.rows, aug.columns);
        }
    }
}
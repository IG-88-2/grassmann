use crate::{Number, core::matrix::Matrix};





pub fn id<T: Number>(size: usize) -> Matrix<T> {

    let zero = T::from_i32(0).unwrap();
    
    let mut data: Vec<T> = vec![zero; size * size];
    
    for i in 0..size {
        data[(size * i) + i] = T::from_i32(1).unwrap();
    }
    
    Matrix {
        data,
        rows: size,
        columns: size
    }
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn identity() {

        let id: Matrix<f64> = Matrix::id(5);
        
        assert!(id.is_diag(), "\n ID should be diagonal {} \n \n", id);
    }

}
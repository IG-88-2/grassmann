use crate::{Number, core::matrix::Matrix, core::vector::Vector, matrix};



pub fn upper_hessenberg<T: Number>(A: &Matrix<T>) -> Matrix<T> {
        
    assert!(A.is_square(), "upper_hessenberg: A should be square");

    let zero = T::from_f64(0.).unwrap();
    
    let mut K = A.clone();
    
    if K.rows <= 2 {
       return K; 
    }
    
    for i in 0..(K.columns - 1) {
        let l = K.rows - i - 1;
        let mut x = Vector::new(vec![zero; l]);
        let mut ce = Vector::new(vec![zero; l]);

        for j in (i + 1)..K.rows {
            x[j - i - 1] = K[[j, i]];
        }

        ce[0] = T::from_f64( x.length() ).unwrap();
        
        let mut v: Vector<T> = &x - &ce;
    
        v.normalize();

        for j in i..K.columns {
            
            let mut x = Vector::new(vec![zero; v.data.len()]);
            
            for k in 0..v.data.len() {
                x[k] = K[[k + i + 1, j]];
            }
            
            let u = &v * T::from_f64( 2. * (&v * &x) ).unwrap();
            
            for k in 0..u.data.len() {
                K[[k + i + 1, j]] -= u[k];
            }
        }

        for j in 0..K.rows {
            
            let mut x = Vector::new(vec![zero; v.data.len()]);
            
            for k in 0..v.data.len() {
                x[k] = K[[j, k + i + 1]];
            }
            
            let u = &v * T::from_f64( 2. * (&v * &x) ).unwrap();
            
            for k in 0..u.data.len() {
                K[[j, k + i + 1]] -= u[k];
            }
        }
    }

    K
}



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };

    #[test]
    fn upper_hessenberg_test() {

        let test = 120;

        for i in 3..test {
            let size = 6;
            let max = 50.;
            let mut A: Matrix<f64> = Matrix::rand(size, size, max);
            let mut H = A.upper_hessenberg();
            
            println!("\n H is {} \n", H);
            assert!(H.is_upper_hessenberg(), "H should be upper hessenberg");
        }
    }

}
use crate::{Number, core::vector::Vector, core::matrix::Matrix};
use super::{ qr::{ form_P } };



pub fn lower_hessenberg<T: Number>(A: &Matrix<T>) -> Matrix<T> {

    assert!(A.is_square(), "lower_hessenberg: A should be square");

    let zero = T::from_f64(0.).unwrap();

    let one = T::from_f64(1.).unwrap();

    let mut H = A.clone();
    
    if H.rows <= 2 {
       return H;
    }

    let f = move |x: T| {
        let y = T::to_i64(&x).unwrap();
        T::from_i64(y).unwrap()
    };

    for i in (2..(H.columns)).rev() {
        let mut x = Vector::new(vec![zero; i]);
        let mut ce = Vector::new(vec![zero; i]);

        for j in 0..i {
            x[j] = H[[j, i]];
        }
        
        ce[i - 1] = T::from_f64( x.length() ).unwrap();
        
        let mut v: Vector<T> = &x - &ce;
    
        v.normalize();

        let P = form_P(&v, H.rows, false);
        
        H = &(&P * &H) * &P;
    }

    H
}



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };

    

    #[test]
    fn lower_hessenberg_test() {

        let test = 15;

        for i in 3..test {
            let size = i;
            let max = 50.;
            let mut A: Matrix<f64> = Matrix::rand(size, size, max);
            let mut H = A.lower_hessenberg();
            
            println!("\n H is {} \n", H);
            
            assert!(H.is_lower_hessenberg(), "H should be lower hessenberg");
        }
    }
}
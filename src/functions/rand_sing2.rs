use std::cmp::min;
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use crate::{Number, core::matrix::Matrix, core::vector::Vector};



pub fn rand_sing2<T: Number>(size: usize, n: usize, max: f64) -> Matrix<T> {
        
    assert!(n < size, "rand_sing2: n < size");

    let mut rng = rand::thread_rng();

    let zero = T::from_f64(0.).unwrap();

    let span = size - n;

    let mut basis: Vec<Vector<T>> = Vec::new();

    let mut null: Vec<Vector<T>> = Vec::new(); 

    for i in 0..span {
        let v: Vector<T> = Vector::rand(size as u32, max);
        basis.push(v);
    }

    for i in 0..n {
        let mut v: Vector<T> = Vector::new(vec![zero; size]);

        for j in 0..basis.len() {
            let mut k = basis[j].clone();
            let s: f64 = rng.gen_range(0., 1.);
            k = k * T::from_f64(s).unwrap();
            v = v + k;
        }

        null.push(v);
    }
    
    basis.append(&mut null);

    let result = Matrix::from_basis(basis);

    let P: Matrix<T> = Matrix::rand_perm(result.columns);

    result * P
}



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };
    use rand::prelude::*;
    use rand::Rng;
    
    

    #[test]
    fn rand_sing2_test() {

        let test = 30;

        for i in 2..test {

            let size = i;

            let max = 5.;
            
            let mut rng = rand::thread_rng();
            
            let n: u32 = rng.gen_range(0, size - 1);
            
            let A: Matrix<f64> = Matrix::rand_sing2(size as usize, n as usize, max);
    
            println!("\n A rank is {}, n is {}, size is {} \n", A.rank(), n, size);

            assert!(A.is_square(), "A is square");

            assert_eq!(A.rank() + n, size, "A.rank() + n = size");
        }
    }

}
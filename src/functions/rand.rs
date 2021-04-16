use crate::{Number, core::matrix::Matrix};
use rand::prelude::*;
use rand::Rng;



pub fn rand<T: Number>(rows: usize, columns: usize, max: f64) -> Matrix<T> {

    let mut A = Matrix::new(rows, columns);

    let mut rng = rand::thread_rng();

    for i in 0..columns {
        for j in 0..rows {
            let value: f64 = rng.gen_range(-max, max);
            let value = ( value * 100. ).round() / 100.;
            A[[j,i]] = T::from_f64(value).unwrap();
        }
    }

    A
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };
    use rand::prelude::*;
    use rand::Rng;
    
    #[test] 
    fn rand() {

        let max = 100000.;

        let mut rng = rand::thread_rng();
        
        let value: f64 = rng.gen_range(-max, max);

        let d = ( value * 10000. ).round() / 10000.;

        println!("\n value is {} | {} \n", value, d);

    }

}
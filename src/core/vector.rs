use std::{
    fmt,
    fmt::{
        Display, 
        Formatter
    }, 
    ops::{
        Add, 
        AddAssign, 
        Index, 
        IndexMut,
        Sub,
        SubAssign,
        Mul,
        MulAssign,
        Div,
        Neg
    }
};
use num_traits::{ Num, identities, cast };

use crate::matrix::Number;
//TODO move traits in main

#[macro_export]
macro_rules! vector {
    (
        $($x:expr),+
    ) => {
        {
            Vector::new(vec![
                $($x),+
            ])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Vector <T> {
    pub data: Vec<T>
}

impl <T: Number> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{:?}, {:?}, {:?}, {:?}] \n", 
            self[0], self[1], self[2], self[3]
        )
    }
}

impl <T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, idx:usize) -> &T {
        &self.data[idx]
    }
}



impl <T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, idx:usize) -> &mut T {
        &mut self.data[idx]
    }
}



impl <T> Vector <T> {

    pub fn new(data: Vec<T>) -> Vector<T> {
        Vector {
            data
        }
    }
    
    fn apply(&mut self, f: fn(f64) -> f64) {
        
    }
}



impl <T> Add for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, b:&Vector<T>) -> Vector<T> {
        Vector::new(vec![
            //self[0] + b[0], self[1] + b[1], self[2] + b[2], self[3] + b[3]
        ])
    }
}



impl <T> AddAssign for Vector<T> {
    fn add_assign(&mut self, b:Vector<T>) {
        
    }
}



impl <T> Sub for &Vector<T> {
    type Output = Vector<T>;

    fn sub(self, b:&Vector<T>) -> Vector<T> {
        Vector::new(vec![
            //self[0] - b[0], self[1] - b[1], self[2] - b[2], self[3] - b[3]
        ])
    }
}



impl <T> SubAssign for Vector<T> {
    fn sub_assign(&mut self, b:Vector<T>) {
        
    }
}



impl <T: Number> Mul for &Vector<T> {
    type Output = T;

    fn mul(self, b: &Vector<T>) -> T {
        T::from_f64(1.).unwrap()
    }
}



impl <T: Number> Mul <T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(mut self, s:T) -> Vector<T> {
        
        self
    }
}



impl <T: Number> Div <T> for Vector<T> {
    type Output = Vector<T>;
    
    fn div(mut self, s:T) -> Vector<T> {
        self
    }
}



impl <T: Number> PartialEq for Vector<T> {
    fn eq(&self, b: &Vector<T>) -> bool {
        self[0] == b[0] &&
        self[1] == b[1] &&
        self[2] == b[2] &&
        self[3] == b[3]
    }
}



impl <T: Number> Eq for Vector<T> {}



impl <T: Number>  Neg for Vector<T> {

    type Output = Self;
    
    fn neg(mut self) -> Self {
        self
    }
}



mod tests {
    use std::f64::consts::PI;

    use super::{Number, Vector};
    
    #[test]
    fn operations() {
        
        /*let p: Vector<i32> = vector![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ];*/

        let y: Vec<f32> = Vec::new();

        let k = Vector::new(
            y
        );


        //assert_eq!(1, 2, "hey {}", k);
        
    }
}

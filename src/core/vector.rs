use std::{fmt, fmt::{
        Display, 
        Formatter
    }, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};
use num_traits::{identities,pow};
use crate::{Float, Number};
use super::{vector3::Vector3, vector4::Vector4, utils::eq_eps_f64};
use rand::prelude::*;
use rand::Rng;



#[macro_export]
macro_rules! vector {
    (
        $($x:expr),*
    ) => {
        {
            Vector::new(vec![
                $($x),*
            ])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Vector <T:Number> {
    pub data: Vec<T>
}



impl <T: Number> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        
        let v: Vec<String> = self.data.clone().iter_mut().map(|x| { x.to_string() }).collect();

        write!(f, "\n [{}] \n", v.join(", "))
    }
}



impl <T:Number> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, idx:usize) -> &T {
        &self.data[idx]
    }
}



impl <T:Number> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, idx:usize) -> &mut T {
        &mut self.data[idx]
    }
}



impl <T:Number> Vector <T> {

    pub fn new(data: Vec<T>) -> Vector<T> {
        Vector {
            data
        }
    }
    

    
    pub fn length(&self) -> f64 {
        
        let result: T = self.data.iter().fold(identities::zero(), |mut sum: T, val: &T| { sum += pow(*val, 2); sum });
       
        T::to_f64(&result).unwrap().sqrt()
    }



    pub fn normalize(&mut self) {

        let l = self.length();

        if l != 0. {
            *self /= T::from_f64(l).unwrap();
        }
    }
    


    pub fn apply(&mut self, f: fn(&T) -> T) {

        self.data = self.data.iter().map(|x| { f(x) }).collect();
    }



    pub fn rand(length: u32, max: Float) -> Vector<T> {

        let mut v: Vector<T> = vector![];

        let mut rng = rand::thread_rng();

        for _i in 0..length {
            let x: f64 = rng.gen_range(0., max); 
            let x = T::from_f64(x).unwrap();
            v.data.push(x);
        }

        v
    }



    pub fn zeros(l:usize) -> Vector<T> {
        let zero = T::from_f64(0.).unwrap();
        let v = vec![zero;l];
        Vector::new(v)
    }



    pub fn ones(l:usize) -> Vector<T> {
        let one = T::from_f64(1.).unwrap();
        let v = vec![one;l];
        Vector::new(v)
    }

    

    pub fn is_zero(&self) -> bool {
        
        self.data.iter().all(|x| { 

            (T::to_f64(x).unwrap()).abs() < f32::EPSILON as f64 

        })
    } 
}



fn add<T:Number>(a:&Vector<T>, b:&Vector<T>) -> Vector<T> {

    let zero = identities::zero();

    let mut v: Vector<T> = Vector::new(vec![zero; a.data.len()]);

    for i in 0..a.data.len() {
        v[i] = a[i] + b[i];
    }

    v
}



fn sub<T:Number>(a:&Vector<T>, b:&Vector<T>) -> Vector<T> {

    let zero = identities::zero();

    let mut v: Vector<T> = Vector::new(vec![zero; a.data.len()]);

    for i in 0..a.data.len() {
        v[i] = a[i] - b[i];
    }

    v
}



fn mul<T:Number>(a:&Vector<T>, b:&Vector<T>) -> Option<Float> {

    let mut acc: T = identities::zero();

    if a.data.len() != b.data.len() {
        return None;
    }

    for i in 0..a.data.len() {
        acc += a[i] * b[i];
    }

    T::to_f64(&acc)
}



fn scale<T:Number>(a:&Vector<T>, s:T) -> Vector<T> {
    
    let zero = T::from_f64(0.).unwrap();

    let mut v = Vector::new(vec![zero; a.data.len()]); //TODO vector![];

    for i in 0..a.data.len() {
        v[i] = a[i] * s; 
    }

    v
}



impl <T:Number> Add for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, b:&Vector<T>) -> Vector<T> {
        add(&self, b)
    }
}



impl <T:Number> Add for Vector<T> {
    type Output = Vector<T>;

    fn add(self, b:Vector<T>) -> Vector<T> {
        add(&self, &b)
    }
}



impl <T:Number> Sub for &Vector<T> {
    type Output = Vector<T>;

    fn sub(self, b:&Vector<T>) -> Vector<T> {
        sub(&self, b)
    }
}



impl <T:Number> Sub for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, b:Vector<T>) -> Vector<T> {
        sub(&self, &b)
    }
}



impl <T:Number> Mul for &Vector<T> {
    type Output = Float;

    fn mul(self, b:&Vector<T>) -> Float  {
        let result = mul(&self, b);
        result.unwrap()
    }
}



impl <T:Number> Mul for Vector<T> {
    type Output = Float;

    fn mul(self, b:Vector<T>) -> Float  {
        let result = mul(&self, &b);
        result.unwrap()
    }
}



impl <T:Number> Mul <T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, s:T) -> Vector<T> {
        scale(&self, s)
    }
}



impl <T:Number> Mul <T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, s:T) -> Vector<T> {
        scale(&self, s)
    }
}



impl <T:Number> Div <T> for &Vector<T> {
    type Output = Vector<T>;
    
    fn div(self, s:T) -> Vector<T> {
        let one: T = identities::one();
        let c = one / s;
        scale(&self, c)
    }
}



impl <T:Number> Div <T> for Vector<T> {
    type Output = Vector<T>;
    
    fn div(self, s:T) -> Vector<T> {
        let one: T = identities::one();
        let c = one / s;
        scale(&self, c)
    }
}



impl <T:Number> AddAssign for Vector<T> {
    fn add_assign(&mut self, b:Vector<T>) {
        for i in 0..self.data.len() {
            self[i] = self[i] + b[i];
        }
    }
}



impl <T:Number> SubAssign for Vector<T> {
    fn sub_assign(&mut self, b:Vector<T>) {
        for i in 0..self.data.len() {
            self[i] = self[i] - b[i];
        }
    }
}



impl <T:Number> MulAssign <T> for Vector<T> {
    fn mul_assign(&mut self, s:T)  {
        for i in 0..self.data.len() {
            self[i] = self[i] * s; 
        }
    }
}



impl <T:Number> DivAssign <T> for Vector<T> {
    fn div_assign(&mut self, s:T)  {
        for i in 0..self.data.len() {
            self[i] = self[i] / s; 
        }
    }
}



impl <T:Number> PartialEq for Vector<T> {
    fn eq(&self, b: &Vector<T>) -> bool {

        if self.data.len() != b.data.len() {
           return false;
        }

        for i in 0..self.data.len() {
            if self[i] != b[i] {
               return false;
            }
        }

        true
    }
}



impl <T:Number> Eq for Vector<T> {}



impl <T:Number> Neg for Vector<T> {
    type Output = Vector<T>;
    
    fn neg(self) -> Vector<T> {
        let c = T::from_i32(-1).unwrap();
        scale(&self, c)
    }
}



impl <T:Number> From<&Vector3> for Vector<T> {
    fn from(v: &Vector3) -> Vector<T> {
        vector![
            T::from_f64(v.x).unwrap(),
            T::from_f64(v.y).unwrap(),
            T::from_f64(v.z).unwrap()
        ]
    }
}



impl <T:Number> From<Vector3> for Vector<T> {
    fn from(v: Vector3) -> Vector<T> {
        vector![
            T::from_f64(v.x).unwrap(),
            T::from_f64(v.y).unwrap(),
            T::from_f64(v.z).unwrap()
        ]
    }
}



impl <T:Number> From<&Vector4> for Vector<T> {
    fn from(v: &Vector4) -> Vector<T> {
        vector![
            T::from_f64(v.x).unwrap(),
            T::from_f64(v.y).unwrap(),
            T::from_f64(v.z).unwrap(),
            T::from_f64(v.t).unwrap()
        ]
    }
}



impl <T:Number> From<Vector4> for Vector<T> {
    fn from(v: Vector4) -> Vector<T> {
        vector![
            T::from_f64(v.x).unwrap(),
            T::from_f64(v.y).unwrap(),
            T::from_f64(v.z).unwrap(),
            T::from_f64(v.t).unwrap()
        ]
    }
}



mod tests {
    use std::f64::consts::PI;
    use super::{
        Vector,
        eq_eps_f64
    };



    #[test]
    fn length() {

        let x = vector![1., 1., 1., 1.];

        let l = x.length();
        
        assert_eq!(2., l, "length should equal 3 {}", l);
    }



    #[test]
    fn normalize() {

        let mut x = vector![77.,34.,2.5,8.9];

        x.normalize();

        let l = x.length();
        
        assert!(eq_eps_f64(l, 1.), "length should equal 1 {}", l);
    }
}

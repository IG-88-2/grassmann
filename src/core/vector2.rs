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
use crate::Float;



#[macro_export]
macro_rules! vec2 {
    ($x:expr, $y:expr) => {
        {
            Vector2::new([$x,$y])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Vector2 {
    pub data: [Float; 2]
}

impl Display for Vector2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n [{}, {}] \n", self[0], self[1])
    }
}

impl Index<usize> for Vector2 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Vector2 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Vector2 {

    pub fn new(data: [Float; 2]) -> Vector2 {
        Vector2 {
            data
        }
    }
    
    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
    }

    pub fn normalize(&mut self) {

        let s = self.data[0] + self.data[1];
        
        if s == 0. {
            return;
        }
        
        let length = (
            self.data[0] * self.data[0] +
            self.data[1] * self.data[1]
        ).sqrt();

        self.data[0] /= length;
        self.data[1] /= length;
    }
}



impl Add for &Vector2 {
    type Output = Vector2;

    fn add(self, b:&Vector2) -> Vector2 {
        Vector2::new([self[0] + b[0], self[1] + b[1]])
    }
}



impl AddAssign for Vector2 {
    fn add_assign(&mut self, b:Vector2) {
        self[0] += b[0];
        self[1] += b[1];
    }
}



impl Sub for &Vector2 {
    type Output = Vector2;

    fn sub(self, b:&Vector2) -> Vector2 {
        Vector2::new([self[0] - b[0], self[1] - b[1]])
    }
}



impl SubAssign for Vector2 {
    fn sub_assign(&mut self, b:Vector2) {
        self[0] -= b[0];
        self[1] -= b[1];
    }
}



impl Mul for &Vector2 {
    type Output = Float;

    fn mul(self, b: &Vector2) -> Float {
        self[0] * b[1] + self[1] * b[1]
    }
}



impl Mul <Float> for Vector2 {
    type Output = Vector2;

    fn mul(mut self, s:f64) -> Vector2 {
        self[0] *= s;
        self[1] *= s;
        self
    }
}



impl Div <Float> for Vector2 {
    type Output = Vector2;
    
    fn div(mut self, s:f64) -> Vector2 {
        if s==0. {
            return self;
        }
        self[0] /= s;
        self[1] /= s;
        self
    }
}



impl PartialEq for Vector2 {
    fn eq(&self, b: &Vector2) -> bool {
        self[0] == b[0] && self[1] == b[1]
    }
}



impl Eq for Vector2 {}



impl Neg for Vector2 {

    type Output = Vector2;
    
    fn neg(mut self) -> Self {
        self[0] *= -1.;
        self[1] *= -1.;
        self[2] *= -1.; 
        self[3] *= -1.;
        self
    }
}



mod tests {
    use std::f64::consts::PI;

    use super::{
        Vector2
    };
    
    #[test]
    fn operations() {
        

        
    }
}

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
macro_rules! vec3 {
    (
        $x:expr, $y:expr, $z:expr
    ) => {
        {
            Vector3::new([$x, $y, $z])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Vector3 {
    pub data: [Float; 3]
}

impl Display for Vector3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}] \n",
            self[0], self[1], self[2]
        )
    }
}

impl Index<usize> for Vector3 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Vector3 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Vector3 {

    pub fn new(data: [Float; 3]) -> Vector3 {
        Vector3 {
            data
        }
    }
    
    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
        self[2] = f(self[2]);
    }
}



impl Add for &Vector3 {
    type Output = Vector3;

    fn add(self, b:&Vector3) -> Vector3 {
        Vector3::new([
            self[0] + b[0], self[1] + b[1], self[2] + b[2]
        ])
    }
}



impl AddAssign for Vector3 {
    fn add_assign(&mut self, b:Vector3) {
        self[0] += b[0];
        self[1] += b[1];
        self[2] += b[2];
    }
}



impl Sub for &Vector3 {
    type Output = Vector3;

    fn sub(self, b:&Vector3) -> Vector3 {
        Vector3::new([
            self[0] - b[0], self[1] - b[1], self[2] - b[2]
        ])
    }
}



impl SubAssign for Vector3 {
    fn sub_assign(&mut self, b:Vector3) {
        self[0] -= b[0];
        self[1] -= b[1];
        self[2] -= b[2];
    }
}



impl Mul for &Vector3 {
    type Output = Float;

    fn mul(self, b: &Vector3) -> Float  {
        self[0] * b[0] + self[1] * b[1] + self[2] * b[2]
    }
}



impl Mul <Float> for Vector3 {
    type Output = Vector3;

    fn mul(mut self, s:f64) -> Vector3 {
        self[0] *= s;
        self[1] *= s;
        self[2] *= s;
        self
    }
}



impl Div <Float> for Vector3 {
    type Output = Vector3;
    
    fn div(mut self, s:f64) -> Vector3 {
        self[0] /= s;
        self[1] /= s;
        self[2] /= s;
        self
    }
}



impl PartialEq for Vector3 {
    fn eq(&self, b: &Vector3) -> bool {
        self[0] == b[0] &&
        self[1] == b[1] &&
        self[2] == b[2]
    }
}



impl Eq for Vector3 {}



impl Neg for Vector3 {

    type Output = Vector3;
    
    fn neg(mut self) -> Self {
        self[0] *= -1.;
        self[1] *= -1.;
        self[2] *= -1.;
        self
    }
}



mod tests {
    use std::f64::consts::PI;

    use super::{
        Vector3
    };
    
    #[test]
    fn operations() {
        

        
    }
}

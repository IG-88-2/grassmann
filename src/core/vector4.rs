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
use crate::{
    Float
};



#[macro_export]
macro_rules! vec4 {
    (
        $x:expr, $y:expr, $z:expr, $t:expr
    ) => {
        {
            Vector4::new([
                $x, $y, $z, $t
            ])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Vector4 {
    pub data: [Float; 4]
}

impl Display for Vector4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}, {}] \n", 
            self[0], self[1], self[2], self[3]
        )
    }
}

impl Index<usize> for Vector4 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Vector4 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Vector4 {

    pub fn new(data: [Float; 4]) -> Vector4 {
        Vector4 {
            data
        }
    }



    pub fn cross(&mut self, b: Vector4) {

    }



    pub fn normalize(&mut self) {

        let s = self.data[0] + self.data[1] + self.data[2] + self.data[3];
        
        if s == 0. {
            return;
        }
        
        let length = (
            self.data[0] * self.data[0] +
            self.data[1] * self.data[1] +
            self.data[2] * self.data[2] +
            self.data[3] * self.data[3] 
        ).sqrt();

        self.data[0] /= length;
        self.data[1] /= length;
        self.data[2] /= length;
        self.data[3] /= length;
    }

    
    
    fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
        self[2] = f(self[2]); 
        self[3] = f(self[3]);
        self[4] = f(self[4]);
    }
}



impl Add for &Vector4 {
    type Output = Vector4;

    fn add(self, b:&Vector4) -> Vector4 {
        Vector4::new([
            self[0] + b[0], self[1] + b[1], self[2] + b[2], self[3] + b[3]
        ])
    }
}



impl AddAssign for Vector4 {
    fn add_assign(&mut self, b:Vector4) {
        self[0] += b[0];
        self[1] += b[1];
        self[2] += b[2];
        self[3] += b[3];
    }
}



impl Sub for &Vector4 {
    type Output = Vector4;

    fn sub(self, b:&Vector4) -> Vector4 {
        Vector4::new([
            self[0] - b[0], self[1] - b[1], self[2] - b[2], self[3] - b[3]
        ])
    }
}



impl SubAssign for Vector4 {
    fn sub_assign(&mut self, b:Vector4) {
        self[0] -= b[0];
        self[1] -= b[1];
        self[2] -= b[2];
        self[3] -= b[3];
    }
}



impl Mul for &Vector4 {
    type Output = Float;

    fn mul(self, b: &Vector4) -> Float {
        self[0] * b[0] + self[1] * b[1] + self[2] * b[2] + self[3] * b[3]
    }
}



impl Mul <Float> for Vector4 {
    type Output = Vector4;

    fn mul(mut self, s:f64) -> Vector4 {
        self[0] *= s;
        self[1] *= s;
        self[2] *= s;
        self[3] *= s;
        self
    }
}



impl Div <Float> for Vector4 {
    type Output = Vector4;
    
    fn div(mut self, s:f64) -> Vector4 {
        self[0] /= s;
        self[1] /= s;
        self[2] /= s;
        self[3] /= s;
        self
    }
}



impl PartialEq for Vector4 {
    fn eq(&self, b: &Vector4) -> bool {
        self[0] == b[0] &&
        self[1] == b[1] &&
        self[2] == b[2] &&
        self[3] == b[3]
    }
}



impl Eq for Vector4 {}



impl Neg for Vector4 {

    type Output = Vector4;
    
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
        Vector4
    };
    
    #[test]
    fn operations() {
        

        
    }
}

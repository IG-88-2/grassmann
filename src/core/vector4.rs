use std::{fmt, fmt::{
        Display, 
        Formatter
    }, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};
use crate::{
    Float
};

use super::vector3::Vector3;



#[macro_export]
macro_rules! vec4 {
    (
        $x:expr, $y:expr, $z:expr, $t:expr
    ) => {
        {
            Vector4::new($x, $y, $z, $t)
        }
    };
}



#[derive(Clone, Copy, Debug)]
pub struct Vector4 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
    pub t: Float
}



impl Display for Vector4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}, {}] \n", 
            self.x, self.y, self.z, self.t
        )
    }
}



impl Index<usize> for Vector4 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        match idx {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.t
        }
    }
}



impl IndexMut<usize> for Vector4 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        match idx {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.t
        }
    }
}



impl Vector4 {

    pub fn new(x: Float, y:Float, z:Float, t:Float) -> Vector4 {
        Vector4 {
            x,
            y,
            z,
            t
        }
    }



    pub fn length(&self) -> Float {
        (self.x.powf(2.) + self.y.powf(2.) + self.z.powf(2.) + self.t.powf(2.)).sqrt()
    }



    pub fn normalize(&mut self) {

        let s = self.x + self.y + self.z;
        
        if s == 0. {
            return;
        }
        
        *self /= self.length();
    }
    


    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self.x = f(self.x);
        self.y = f(self.y);
        self.z = f(self.z);
        self.t = f(self.t);
    }
}



fn add(a:&Vector4, b:&Vector4) -> Vector4 {
    Vector4::new(
        a.x + b.x,
        a.y + b.y, 
        a.z + b.z,
        a.t + b.t
    )
}



fn sub(a:&Vector4, b:&Vector4) -> Vector4 {
    Vector4::new(
        a.x - b.x,
        a.y - b.y, 
        a.z - b.z,
        a.t - b.t
    )
}



fn mul(a:&Vector4, b:&Vector4) -> Float {
    a.x * b.x + 
    a.y * b.y + 
    a.z * b.z +
    a.t * b.t
}



fn scale(a:&Vector4, s:f64) -> Vector4 {
    Vector4::new(
        a.x * s,
        a.y * s, 
        a.z * s,
        a.t * s
    )
}



impl Add for &Vector4 {
    type Output = Vector4;

    fn add(self, b:&Vector4) -> Vector4 {
        add(&self, b)
    }
}



impl Add for Vector4 {
    type Output = Vector4;

    fn add(self, b:Vector4) -> Vector4 {
        add(&self, &b)
    }
}



impl Sub for &Vector4 {
    type Output = Vector4;

    fn sub(self, b:&Vector4) -> Vector4 {
        sub(&self, b)
    }
}



impl Sub for Vector4 {
    type Output = Vector4;

    fn sub(self, b:Vector4) -> Vector4 {
        sub(&self, &b)
    }
}



impl Mul for &Vector4 {
    type Output = Float;

    fn mul(self, b:&Vector4) -> Float  {
        mul(&self, b)
    }
}



impl Mul for Vector4 {
    type Output = Float;

    fn mul(self, b:Vector4) -> Float  {
        mul(&self, &b)
    }
}



impl Mul <Float> for &Vector4 {
    type Output = Vector4;

    fn mul(self, s:Float) -> Vector4 {
        scale(&self, s)
    }
}



impl Mul <Float> for Vector4 {
    type Output = Vector4;

    fn mul(self, s:Float) -> Vector4 {
        scale(&self, s)
    }
}



impl Div <Float> for &Vector4 {
    type Output = Vector4;
    
    fn div(self, s:Float) -> Vector4 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl Div <Float> for Vector4 {
    type Output = Vector4;
    
    fn div(self, s:Float) -> Vector4 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl AddAssign for Vector4 {
    fn add_assign(&mut self, b:Vector4) {
        self.x += b.x;
        self.y += b.y;
        self.z += b.z;
    }
}



impl SubAssign for Vector4 {
    fn sub_assign(&mut self, b:Vector4) {
        self.x -= b.x;
        self.y -= b.y;
        self.z -= b.z;
        self.t -= b.t;
    }
}



impl MulAssign for Vector4 {
    fn mul_assign(&mut self, b:Vector4)  {
        self.x *= b.x; 
        self.y *= b.y; 
        self.z *= b.z;
        self.t *= b.t;
    }
}



impl MulAssign <Float> for Vector4 {
    fn mul_assign(&mut self, s:Float)  {
        self.x = self.x * s; 
        self.y = self.y * s; 
        self.z = self.z * s;
        self.t = self.t * s;
    }
}



impl DivAssign <Float> for Vector4 {
    fn div_assign(&mut self, s:Float)  {
        self.x = self.x / s; 
        self.y = self.y / s; 
        self.z = self.z / s; 
        self.t = self.t / s; 
    }
}



impl PartialEq for Vector4 {
    fn eq(&self, b: &Vector4) -> bool {
        self.x == b.x &&
        self.y == b.y &&
        self.z == b.z &&
        self.t == b.t
    }
}



impl Eq for Vector4 {}



impl Neg for Vector4 {
    type Output = Vector4;
    
    fn neg(mut self) -> Vector4 {
        self.x *= -1.;
        self.y *= -1.;
        self.z *= -1.;
        self.t *= -1.;
        self
    }
}



impl From<Vector3> for Vector4 {
    fn from(v: Vector3) -> Vector4 {
        Vector4::new(
            v.x,
            v.y,
            v.z,
            0.
        )
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

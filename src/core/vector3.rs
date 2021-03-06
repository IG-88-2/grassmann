use std::{fmt, fmt::{
        Display, 
        Formatter
    }, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};
use crate::Float;

use super::vector4::Vector4;



#[macro_export]
macro_rules! vec3 {
    (
        $x:expr, $y:expr, $z:expr
    ) => {
        {
            Vector3::new($x, $y, $z)
        }
    };
}



#[derive(Clone, Copy, Debug)]
pub struct Vector3 {
    pub x: Float,
    pub y: Float,
    pub z: Float
}



impl Index<usize> for Vector3 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        match idx {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z
        }
    }
}



impl IndexMut<usize> for Vector3 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        match idx {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z
        }
    }
}



impl Display for Vector3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}] \n",
            self.x, self.y, self.z
        )
    }
}



impl Vector3 {

    pub fn new(x: Float, y:Float, z:Float) -> Vector3 {
        Vector3 {
            x,
            y,
            z
        }
    }



    pub fn length(&self) -> Float {
        (self.x.powf(2.) + self.y.powf(2.) + self.z.powf(2.)).sqrt()
    }



    pub fn normalize(&mut self) {

        let s = self.x + self.y + self.z;
        
        if s == 0. {
            return;
        }
        
        *self /= self.length();
    }
    


    pub fn cross(&self, b: &Vector3) -> Vector3 {

        Vector3::new(
            self.y * b.z - self.z * b.y,
            self.z + b.x - self.x * b.z, 
            self.x + b.y - self.y * b.x 
        )
    }



    pub fn distance(&self, b: &Vector3) -> Float {

        let v =  Vector3::new(
            self.x - b.x,
            self.y - b.y, 
            self.z - b.z
        );

        v.length()
    }



    pub fn angle(&self, b: &Vector3) -> Float {

        let mut d: Float = self * b;

        let m = self.length() * b.length();

        if m == 0. {
            return 0.
        }

        d /= m;

        d.acos()
    }



    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self.x = f(self.x);
        self.y = f(self.y);
        self.z = f(self.z);
    }
}



fn add(a:&Vector3, b:&Vector3) -> Vector3 {
    Vector3::new(
        a.x + b.x,
        a.y + b.y, 
        a.z + b.z 
    )
}



fn sub(a:&Vector3, b:&Vector3) -> Vector3 {
    Vector3::new(
        a.x - b.x,
        a.y - b.y, 
        a.z - b.z 
    )
}



fn mul(a:&Vector3, b:&Vector3) -> Float {
    a.x * b.x + 
    a.y * b.y + 
    a.z * b.z
}



fn scale(a:&Vector3, s:f64) -> Vector3 {
    Vector3::new(
        a.x * s,
        a.y * s, 
        a.z * s
    )
}



impl Add for &Vector3 {
    type Output = Vector3;

    fn add(self, b:&Vector3) -> Vector3 {
        add(&self, b)
    }
}



impl Add for Vector3 {
    type Output = Vector3;

    fn add(self, b:Vector3) -> Vector3 {
        add(&self, &b)
    }
}



impl Sub for &Vector3 {
    type Output = Vector3;

    fn sub(self, b:&Vector3) -> Vector3 {
        sub(&self, b)
    }
}



impl Sub for Vector3 {
    type Output = Vector3;

    fn sub(self, b:Vector3) -> Vector3 {
        sub(&self, &b)
    }
}



impl Mul for &Vector3 {
    type Output = Float;

    fn mul(self, b:&Vector3) -> Float  {
        mul(&self, b)
    }
}



impl Mul for Vector3 {
    type Output = Float;

    fn mul(self, b:Vector3) -> Float  {
        mul(&self, &b)
    }
}



impl Mul <Float> for &Vector3 {
    type Output = Vector3;

    fn mul(self, s:Float) -> Vector3 {
        scale(&self, s)
    }
}



impl Mul <Float> for Vector3 {
    type Output = Vector3;

    fn mul(self, s:Float) -> Vector3 {
        scale(&self, s)
    }
}



impl Div <Float> for &Vector3 {
    type Output = Vector3;
    
    fn div(self, s:Float) -> Vector3 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl Div <Float> for Vector3 {
    type Output = Vector3;
    
    fn div(self, s:Float) -> Vector3 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl AddAssign for Vector3 {
    fn add_assign(&mut self, b:Vector3) {
        self.x += b.x;
        self.y += b.y;
        self.z += b.z;
    }
}



impl SubAssign for Vector3 {
    fn sub_assign(&mut self, b:Vector3) {
        self.x -= b.x;
        self.y -= b.y;
        self.z -= b.z;
    }
}



impl MulAssign for Vector3 {
    fn mul_assign(&mut self, b:Vector3)  {
        self.x *= b.x; 
        self.y *= b.y; 
        self.z *= b.z;
    }
}



impl MulAssign <Float> for Vector3 {
    fn mul_assign(&mut self, s:Float)  {
        self.x = self.x * s; 
        self.y = self.y * s; 
        self.z = self.z * s; 
    }
}



impl DivAssign <Float> for Vector3 {
    fn div_assign(&mut self, s:Float)  {
        self.x = self.x / s; 
        self.y = self.y / s; 
        self.z = self.z / s; 
    }
}



impl PartialEq for Vector3 {
    fn eq(&self, b: &Vector3) -> bool {
        self.x == b.x &&
        self.y == b.y &&
        self.z == b.z
    }
}



impl Eq for Vector3 {}



impl Neg for Vector3 {
    type Output = Vector3;
    
    fn neg(mut self) -> Vector3 {
        self.x *= -1.;
        self.y *= -1.;
        self.z *= -1.;
        self
    }
}



impl From<Vector4> for Vector3 {
    fn from(v: Vector4) -> Vector3 {
        Vector3::new(
            v.x,
            v.y,
            v.z
        )
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

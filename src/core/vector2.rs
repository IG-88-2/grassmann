use std::{f32::EPSILON, f64::consts::PI, fmt, fmt::{
    Display, 
    Formatter
}, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};
use crate::{Float, functions::utils::{clamp, eq_eps_f64}};
use super::{matrix2::Matrix2, matrix3::Matrix3, matrix4::Matrix4, vector::Vector, vector3::Vector3, vector4::Vector4};
use rand::prelude::*;
use rand::Rng;



#[macro_export]
macro_rules! vec2 {
    (
        $x:expr, $y:expr
    ) => {
        {
            Vector2::new($x, $y)
        }
    };
}



#[derive(Clone, Copy, Debug)]
pub struct Vector2 {
    pub x: Float,
    pub y: Float
}



impl Index<usize> for Vector2 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        match idx {
            0 => &self.x,
            _ => &self.y
        }
    }
}



impl IndexMut<usize> for Vector2 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        match idx {
            0 => &mut self.x,
            _ => &mut self.y
        }
    }
}



impl Display for Vector2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}] \n",
            self.x, self.y
        )
    }
}



impl Vector2 {

    pub fn new(x: Float, y:Float) -> Vector2 {
        Vector2 {
            x,
            y
        }
    }



    pub fn length(&self) -> Float {
        (self.x.powf(2.) + self.y.powf(2.)).sqrt()
    }



    pub fn normalize(&mut self) {

        let s = self.x + self.y;
        
        if s == 0. {
            return;
        }
        
        *self /= self.length();
    }



    pub fn distance(&self, b: &Vector2) -> Float {

        let v = Vector2::new(
            self.x - b.x,
            self.y - b.y
        );

        v.length()
    }



    pub fn angle(&self, b: &Vector2) -> Float {

        let mut d: Float = self * b;

        let m = self.length() * b.length();

        if m == 0. {
           return 0.
        }

        //println!("angle: d is {}, m is {}", d, m);

        d /= m;

        d = clamp(-1., 1.)(d);

        //println!("angle: d is {}", d);

        d.acos()
    }



    pub fn project_on(&self, b:&Vector2) -> Vector2 {

        let s: Float = (b * self) / (b * b);

        b * s
    }



    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self.x = f(self.x);
        self.y = f(self.y);
    }



    pub fn rand(max: Float) -> Vector2 {

        let mut rng = rand::thread_rng();

        let x = rng.gen_range(0., max); 
        let y = rng.gen_range(0., max);

        vec2![x, y]
    }
}



fn add(a:&Vector2, b:&Vector2) -> Vector2 {
    Vector2::new(
        a.x + b.x,
        a.y + b.y
    )
}



fn sub(a:&Vector2, b:&Vector2) -> Vector2 {
    Vector2::new(
        a.x - b.x,
        a.y - b.y
    )
}



fn mul(a:&Vector2, b:&Vector2) -> Float {
    a.x * b.x + 
    a.y * b.y
}



fn scale(a:&Vector2, s:f64) -> Vector2 {
    Vector2::new(
        a.x * s,
        a.y * s
    )
}



impl Add for &Vector2 {
    type Output = Vector2;

    fn add(self, b:&Vector2) -> Vector2 {
        add(&self, b)
    }
}



impl Add for Vector2 {
    type Output = Vector2;

    fn add(self, b:Vector2) -> Vector2 {
        add(&self, &b)
    }
}



impl Sub for &Vector2 {
    type Output = Vector2;

    fn sub(self, b:&Vector2) -> Vector2 {
        sub(&self, b)
    }
}



impl Sub for Vector2 {
    type Output = Vector2;

    fn sub(self, b:Vector2) -> Vector2 {
        sub(&self, &b)
    }
}



impl Mul for &Vector2 {
    type Output = Float;

    fn mul(self, b:&Vector2) -> Float  {
        mul(&self, b)
    }
}



impl Mul for Vector2 {
    type Output = Float;

    fn mul(self, b:Vector2) -> Float  {
        mul(&self, &b)
    }
}



impl Mul <Float> for &Vector2 {
    type Output = Vector2;

    fn mul(self, s:Float) -> Vector2 {
        scale(&self, s)
    }
}



impl Mul <Float> for Vector2 {
    type Output = Vector2;

    fn mul(self, s:Float) -> Vector2 {
        scale(&self, s)
    }
}



impl Div <Float> for &Vector2 {
    type Output = Vector2;

    fn div(self, s:Float) -> Vector2 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl Div <Float> for Vector2 {
    type Output = Vector2;

    fn div(self, s:Float) -> Vector2 {
        let c = 1./s;
        scale(&self, c)
    }
}



impl AddAssign for Vector2 {
    fn add_assign(&mut self, b:Vector2) {
        self.x += b.x;
        self.y += b.y;
    }
}



impl SubAssign for Vector2 {
    fn sub_assign(&mut self, b:Vector2) {
        self.x -= b.x;
        self.y -= b.y;
    }
}



impl MulAssign for Vector2 {
    fn mul_assign(&mut self, b:Vector2)  {
        self.x *= b.x; 
        self.y *= b.y;
    }
}



impl MulAssign <Float> for Vector2 {
    fn mul_assign(&mut self, s:Float)  {
        self.x = self.x * s; 
        self.y = self.y * s;
    }
}



impl DivAssign <Float> for Vector2 {
    fn div_assign(&mut self, s:Float)  {
        self.x = self.x / s; 
        self.y = self.y / s;
    }
}



impl PartialEq for Vector2 {
    fn eq(&self, b: &Vector2) -> bool {
        self.x == b.x &&
        self.y == b.y
    }
}



impl Eq for Vector2 {}



impl Neg for Vector2 {
    type Output = Vector2;

    fn neg(mut self) -> Vector2 {
        self.x *= -1.;
        self.y *= -1.;
        self
    }
}



impl From<Vector4> for Vector2 {
    fn from(v: Vector4) -> Vector2 {
        Vector2::new(
            v.x,
            v.y
        )
    }
}



impl From<Vector<Float>> for Vector2 {
    fn from(v: Vector<Float>) -> Vector2 {
        Vector2::new(
            v[0],
            v[1]
        )
    }
}



mod tests {
    use std::{f32::EPSILON, f64::consts::PI};
    use super::{Vector2, Vector3, Matrix2, Matrix3, Matrix4, Vector4, eq_eps_f64, clamp};
    use crate::{vec4, Float};




}

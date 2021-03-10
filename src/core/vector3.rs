use std::{f32::EPSILON, f64::consts::PI, fmt, fmt::{
        Display, 
        Formatter
    }, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign}};
use crate::Float;
use super::{vector::Vector, vector4::Vector4, matrix3::Matrix3, matrix4::Matrix4, utils::{clamp, eq_eps_f64}};
use rand::prelude::*;
use rand::Rng;



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

        let c = Matrix3::cross(self);

        c * b
    }



    pub fn distance(&self, b: &Vector3) -> Float {

        let v = Vector3::new(
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

        //println!("angle: d is {}, m is {}", d, m);

        d /= m;

        d = clamp(-1., 1.)(d);

        //println!("angle: d is {}", d);

        d.acos()
    }



    pub fn project_on(&self, b:&Vector3) -> Vector3 {

        let s: Float = (b * self) / (b * b);

        b * s
    }



    pub fn retrieve_rotation(&self, origin:f64) -> Matrix4 {

        let [x, y, z] = Matrix3::id().into_basis();

        let mut y_ang = self.angle(&y);

        y_ang -= origin;

        let mut z_ang = self.angle(&z);
        
        z_ang -= origin;

        let r = Matrix4::rotation(0., z_ang, y_ang);
        
        r
    } 



    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self.x = f(self.x);
        self.y = f(self.y);
        self.z = f(self.z);
    }



    pub fn rand(max: Float) -> Vector3 {

        let mut rng = rand::thread_rng();

        let x = rng.gen_range(0., max); 
        let y = rng.gen_range(0., max);
        let z = rng.gen_range(0., max);

        vec3![x, y, z]
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



impl From<Vector<Float>> for Vector3 {
    fn from(v: Vector<Float>) -> Vector3 {
        Vector3::new(
            v[0],
            v[1],
            v[2]
        )
    }
}



mod tests {
    use std::{f32::EPSILON, f64::consts::PI};
    use super::{Vector3, Matrix3, Matrix4, Vector4, eq_eps_f64, clamp};
    use crate::{vec4, Float};

    
    #[test]
    fn length() {
        let id = Matrix3::id();
        let basis: [Vector3; 3] = id.into_basis();

        let x = basis[0].length();
        let y = basis[1].length();
        let z = basis[2].length();

        assert_eq!(1., x, "x is 1 ({})", x);
        assert_eq!(1., y, "y is 1 ({})", y);
        assert_eq!(1., z, "z is 1 ({})", z);
        
        let x = vec3![1., 2., 2.];
        let l = x.length();
        assert_eq!(3., l, "length should equal 3 {}", l);
        
        let mut x = Vector3::rand(10.);
        let l0 = x.length();
        x *= 2.;
        let l1 = x.length(); 
        assert_eq!(l0 * 2., l1, "length should double {} - {}", l0, l1);

        let r = Matrix3::rotation(0.43);
        let mut x = Vector3::rand(10.);
        let mut y = Vector3::rand(10.);
        let mut z = Vector3::rand(10.);

        let lx = x.length();
        let ly = y.length();
        let lz = z.length();

        x = &r * x;
        y = &r * y;
        z = &r * z;

        let l2x = x.length();
        let l2y = y.length();
        let l2z = z.length();
        
        assert!(eq_eps_f64(lx, l2x), "length should stay equal {} - {}", lx, l2x);
        assert!(eq_eps_f64(ly, l2y), "length should stay equal {} - {}", ly, l2y);
        assert!(eq_eps_f64(lz, l2z), "length should stay equal {} - {}", lz, l2z);
    }



    #[test]
    fn normalize() {

        let mut x = vec3![77.,34.,2.5];

        x.normalize();

        let l = x.length();
        
        assert!(eq_eps_f64(l, 1.), "length should equal 1 {}", l);
    }



    #[test]
    fn cross() {
        
        let x = Vector3::rand(10.);
        let y = Vector3::rand(10.);
        let z = x.cross(&y);
        
        let dx = x * z;
        let dy = y * z;

        assert!(eq_eps_f64(dx, 0.), "dot x should equal 0 {}", dx);
        assert!(eq_eps_f64(dy, 0.), "dot y should equal 0 {}", dy);
    }



    #[test]
    fn distance() {
        let x = Vector3::rand(10.);
        let z = x.distance(&x);
        assert_eq!(z, 0., "distance to itself should equal 0 {}", z);
        
        let offset = Vector3::rand(10.);
        let l = offset.length();
        let t = Matrix4::translate(offset.x, offset.y, offset.z, 1.);
        
        let d = Vector3::rand(10.);
        let y = Vector4::from(d);
        let yp = &t * y;
        let y = Vector3::from(y);
        let yp = Vector3::from(yp);
        
        let dst = y.distance(&yp);
        assert!(eq_eps_f64(l, dst), "distance should be equal to length of translation vector dl {} dst {}", l, dst);
    }



    #[test]
    fn angle() {
        let x = Vector3::rand(10.);
        let z = x.angle(&x);
        assert!(eq_eps_f64(z, 0.), "angle with itself should equal 0 {}", z);

        let id = Matrix3::id();
        let basis: [Vector3; 3] = id.into_basis();

        let x = basis[0];
        let y = basis[1];
        
        let ang = x.angle(&y);
        
        assert_eq!(ang, PI / 2., "basis angle {}", ang);

        let d = PI / 2.;
        let test = Matrix4::rotation(d,d,d);

        let test1 = Matrix4::rotation(d, 0.,0.);
        let test2 = Matrix4::rotation(0.,d, 0.);
        let test3 = Matrix4::rotation(0.,0.,d);

        println!("test \n 0 {} \n \n 1 {} \n \n 2 {} \n \n 3 {} \n", test, test1, test2, test3);
    }



    #[test]
    fn project() {
        let x = Vector3::rand(10.);
        let y = Vector3::rand(10.);
        let yp = x.project_on(&y);
        let z = y.angle(&yp);
        let xn: Vector3 = x - yp; 
        let z2: f64 = xn * y;
        assert!(eq_eps_f64(z, 0.), "angle should be zero {}", z);
        assert!(eq_eps_f64(z2, 0.), "dot product should be zero {}", z2);
    }




    #[test]
    fn retrieve_rotation() {
        let id = Matrix3::id();
        let basis = id.into_basis();
        let mut x: Vector3 = basis[0].into();
        let mut xm = -x;

        let c = PI;
        let r: Matrix3 = Matrix4::rotation(0., 0., c).into();
        
        x = &r * x;
        xm = &r * xm;
        
        let a = x.angle(&basis[1].into());
        let b = xm.angle(&basis[1].into());

        println!("\n x angle with y is {} \n", a);
        println!("\n xm angle with y is {} \n", b);
        /*
        for i in 0..11 {
            let mut k = 10 - i;
            if k == 0 {
                k = 1;
            }
            let c = PI / k as f64;
            
            let r: Matrix3 = Matrix4::rotation(0., 0., c).into();
            let r2: Matrix3 = Matrix4::rotation(0., 0., -c).into();
            
            let u = r * x;
            let p = r2 * x;

            let a = u.angle(&basis[1].into());
            let q = p.angle(&basis[1].into());

            //let b = x.angle(&basis[2].into()); //a / (2. * PI)
            
            println!("u angle with y is {} | location {} \n", a / (PI), c);
            //println!("p angle with y is {}\n", q / (PI));

            //println!("\n angle with z is {} \n", b);
        }
        */
        /*
        let mut x = vec3![1.,1.,1.]; //Vector3::rand(10.);

        x.normalize();

        let id = vec4![1., 0., 0., 0.];

        let r = x.retrieve_rotation();

        let x = Vector4::from(x);

        let t = &r * id;
        //let r_inv = r.inv().unwrap();

        //let xi = &r_inv * x;

        assert!(false, "\n x {} \n t {} \n r {}", x, t, r);
        */
    }
}

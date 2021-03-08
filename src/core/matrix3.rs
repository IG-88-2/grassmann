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
        Div,
        Neg
    }
};
use crate::{Float, vec3, vector3::Vector3};

use super::matrix4::Matrix4;



#[macro_export]
macro_rules! matrix3 {
    (
        $x1:expr, $y1:expr, $z1:expr,
        $x2:expr, $y2:expr, $z2:expr,
        $x3:expr, $y3:expr, $z3:expr
    ) => {
        {
            Matrix3::new([
                $x1, $y1, $z1,
                $x2, $y2, $z2,
                $x3, $y3, $z3
            ])
        }
    };
}



#[macro_export]
macro_rules! compose_basis3 {
    (
        $x:expr, 
        $y:expr, 
        $z:expr
    ) => {
        {
            Matrix3::new([
                $x[0], $y[0], $z[0],
                $x[1], $y[1], $z[1],
                $x[2], $y[2], $z[2]
            ])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Matrix3 {
    pub data: [Float; 9]
}



impl Display for Matrix3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}] \n [{}, {}, {}] \n [{}, {}, {}] \n", 
            self[0], self[1], self[2],
            self[3], self[4], self[5],
            self[6], self[7], self[8]
        )
    }
}



impl Index<usize> for Matrix3 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Matrix3 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Matrix3 {

    pub fn new(data: [Float; 9]) -> Matrix3 {
        Matrix3 {
            data
        }
    }



    pub fn id() -> Matrix3 {
        Matrix3::new([
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.
        ])
    }


    
    pub fn t(&mut self) {
        *self = Matrix3::new([
            self[0], self[3], self[6],
            self[1], self[4], self[7],
            self[2], self[5], self[8]
        ]);
    }



    pub fn det(&self) -> Float {
        self[0]*(self[4] * self[8] - self[5] * self[7]) - 
        self[1]*(self[3] * self[8] - self[5] * self[6]) + 
        self[2]*(self[3] * self[7] - self[6] * self[4])
    }



    pub fn inv(&self) -> Option<Matrix3> {
        let s = self.det();

        if s==0. {
           return None;
        }
 
        let mut m = Matrix3::new([
            self[4] * self[8] - self[5] * self[7], 
           -1. * (self[3] * self[8] - self[5] * self[6]),  
            self[3] * self[7] - self[4] * self[6],  
            
           -1. * (self[1] * self[8] - self[2] * self[7]),  
            self[0] * self[8] - self[2] * self[6],  
           -1. * (self[0] * self[7] - self[1] * self[6]),  

            self[1] * self[5] - self[2] * self[4],  
           -1. * (self[0] * self[5] - self[2] * self[3]), 
            self[0] * self[4] - self[1] * self[3]
        ]);

        m = m * (1./s);

        m.t();

        Some(m)
    }



    pub fn rotation(rad: f64) -> Matrix3 {

        let z = matrix3![
            rad.cos(), -rad.sin(), 0.,
            rad.sin(),  rad.cos(), 0.,
            0.,         0.,        1.
        ];

        z
    }



    pub fn scale(s: f64) -> Matrix3 {
        matrix3![
            s,  0., 0.,
            0., s,  0., 
            0., 0., 1.
        ]
    }

    

    pub fn translate(x: f64, y: f64) -> Matrix3 {
        matrix3![
            1., 0., x,
            0., 1., y,
            0., 0., 1.
        ]
    }
    


    pub fn sheer(xy:f64, yx:f64) -> Matrix3 {
        matrix3![
            1., yx, 0.,
            xy, 1., 0.,
            0., 0., 1.
        ]
    }



    pub fn cross(v:&Vector3) -> Matrix3 {
        matrix3![
            0., -v.z, v.y,
            v.z, 0., -v.x,
           -v.y, v.x, 0.
        ]
    }



    pub fn from_basis(
        x: Vector3,
        y: Vector3,
        z: Vector3
    ) -> Matrix3 {
        let r = compose_basis3![
            &x,
            &y,
            &z
        ];

        r
    }



    pub fn into_basis(&self) -> [Vector3; 3] {

        let x = vec3![self[0], self[3], self[6]];
        let y = vec3![self[1], self[4], self[7]];
        let z = vec3![self[2], self[5], self[8]];

        [x,y,z]
    }


    
    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
        self[2] = f(self[2]); 
        self[3] = f(self[3]);
        self[4] = f(self[4]);
        self[5] = f(self[5]);
        self[6] = f(self[6]); 
        self[7] = f(self[7]);
        self[8] = f(self[8]);
    }
}



fn add(a:&Matrix3, b:&Matrix3) -> Matrix3 {
    Matrix3::new([
        a[0] + b[0], a[3] + b[3], a[6] + b[6],
        a[1] + b[1], a[4] + b[4], a[7] + b[7],
        a[2] + b[2], a[5] + b[5], a[8] + b[8]
    ])
}



fn sub(a:&Matrix3, b:&Matrix3) -> Matrix3 {
    Matrix3::new([
        a[0] - b[0], a[3] - b[3], a[6] - b[6],
        a[1] - b[1], a[4] - b[4], a[7] - b[7],
        a[2] - b[2], a[5] - b[5], a[8] - b[8]
    ])
}



fn mul(a:&Matrix3, b:&Matrix3) -> Matrix3 {
    Matrix3::new([
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
    ])
}



fn mul_v(a:&Matrix3, v:&Vector3) -> Vector3 {
    Vector3::new(
        a[0] * v.x + a[1] * v.y + a[2] * v.z,
        a[3] * v.x + a[4] * v.y + a[5] * v.z,
        a[6] * v.x + a[7] * v.y + a[8] * v.z
    )
}



impl Add for &Matrix3 {
    type Output = Matrix3;

    fn add(self, b:&Matrix3) -> Matrix3 {
        add(self, b)
    }
}



impl Add for Matrix3 {
    type Output = Matrix3;

    fn add(self, b:Matrix3) -> Matrix3 {
        add(&self, &b)
    }
}



impl Sub for &Matrix3 {
    type Output = Matrix3;

    fn sub(self, b:&Matrix3) -> Matrix3 {
        sub(self, b)
    }
}



impl Sub for Matrix3 {
    type Output = Matrix3;

    fn sub(self, b:Matrix3) -> Matrix3 {
        sub(&self, &b)
    }
}



impl Mul for &Matrix3 {
    type Output = Matrix3;

    fn mul(self, b: &Matrix3) -> Matrix3 {
        mul(self, b)
    }
}



impl Mul for Matrix3 {
    type Output = Matrix3;

    fn mul(self, b: Matrix3) -> Matrix3 {
        mul(&self, &b)
    }
}



impl Mul <Vector3> for &Matrix3 {
    type Output = Vector3;

    fn mul(self, v:Vector3) -> Vector3 {
        mul_v(&self, &v)
    }
}



impl Mul <Vector3> for Matrix3 {
    type Output = Vector3;

    fn mul(self, v:Vector3) -> Vector3 {
        mul_v(&self, &v)
    }
}



impl Mul <&Vector3> for Matrix3 {
    type Output = Vector3;

    fn mul(self, v:&Vector3) -> Vector3 {
        mul_v(&self, v)
    }
}



impl Mul <&Vector3> for &Matrix3 {
    type Output = Vector3;

    fn mul(self, v:&Vector3) -> Vector3 {
        mul_v(self, v)
    }
}



impl Mul <Float> for Matrix3 {
    type Output = Matrix3;

    fn mul(mut self, s:f64) -> Matrix3 {
        self[0] *= s;
        self[1] *= s;
        self[2] *= s;
        self[3] *= s;
        self[4] *= s;
        self[5] *= s;
        self[6] *= s;
        self[7] *= s;
        self[8] *= s;
        self
    }
}



impl Div <Float> for Matrix3 {
    type Output = Matrix3;
    
    fn div(mut self, s:f64) -> Matrix3 {
        self[0] /= s;
        self[1] /= s;
        self[2] /= s;
        self[3] /= s;
        self[4] /= s;
        self[5] /= s;
        self[6] /= s;
        self[7] /= s;
        self[8] /= s;
        self
    }
}



impl AddAssign for Matrix3 {
    fn add_assign(&mut self, b:Matrix3) {
        self[0] += b[0];
        self[1] += b[1];
        self[2] += b[2];
        self[3] += b[3];
        self[4] += b[4];
        self[5] += b[5];
        self[6] += b[6];
        self[7] += b[7];
        self[8] += b[8];
    }
}



impl SubAssign for Matrix3 {
    fn sub_assign(&mut self, b:Matrix3) {
        self[0] -= b[0];
        self[1] -= b[1];
        self[2] -= b[2];
        self[3] -= b[3];
        self[4] -= b[4];
        self[5] -= b[5];
        self[6] -= b[6];
        self[7] -= b[7];
        self[8] -= b[8];
    }
}



impl PartialEq for Matrix3 {
    fn eq(&self, b: &Matrix3) -> bool {
        self[0] == b[0] &&
        self[1] == b[1] &&
        self[2] == b[2] &&
        self[3] == b[3] &&
        self[4] == b[4] &&
        self[5] == b[5] &&
        self[6] == b[6] &&
        self[7] == b[7] &&
        self[8] == b[8]
    }
}



impl Eq for Matrix3 {}



impl Neg for Matrix3 {

    type Output = Matrix3;
    
    fn neg(mut self) -> Self {
        self[0] *= -1.;
        self[1] *= -1.;
        self[2] *= -1.;
        self[3] *= -1.;
        self[4] *= -1.;
        self[5] *= -1.;
        self[6] *= -1.;
        self[7] *= -1.;
        self[8] *= -1.;
        self
    }
}



impl From<Matrix4> for Matrix3 {
    fn from(m: Matrix4) -> Matrix3 {
        matrix3![
            m[0], m[1], m[2],
            m[4], m[5], m[6],
            m[8], m[9], m[10]
        ]
    }
}



mod tests {
    use std::f64::consts::PI;

    use super::{
        Matrix3, Vector3
    };
    
    #[test]
    fn operations() {
        
        let id = Matrix3::id();

        //let k: Vec<Vector3> = id.clone().into();

        let inv = id.inv().unwrap();

        let test = matrix3![
            1., 2., 2.,
            2., 1., 2.,
            1., 2., 3.
        ];
        
        //assert_eq!(1, 0, "test {}", test.det());

        let test_inv = test.inv().unwrap();
        
        let p = &test_inv * &test;
        
        //assert_eq!(1, 2, "id {}", p);
        
    }
}

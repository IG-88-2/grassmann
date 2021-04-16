use std::{f64::consts::PI, fmt, fmt::{
    Display, 
    Formatter
}, ops::{
    Add, 
    AddAssign, 
    Index, 
    IndexMut,
    Sub,
    SubAssign,
    Mul,
    Div,
    Neg
}};
use crate::{Float, functions::utils::eq_eps_f64, matrix, vec3, vec2, vector, vector3::Vector3};
use super::{matrix::Matrix, matrix4::Matrix4, vector::Vector, vector2::Vector2, vector4::Vector4};



#[macro_export]
macro_rules! matrix2 {
    (
        $x1:expr, $y1:expr,
        $x2:expr, $y2:expr
    ) => {
        {
            Matrix2::new([
                $x1, $y1,
                $x2, $y2
            ])
        }
    };
}



#[macro_export]
macro_rules! compose_basis2 {
    (
        $x:expr, 
        $y:expr
    ) => {
        {
            Matrix2::new([
                $x[0], $y[0],
                $x[1], $y[1]
            ])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Matrix2 {
    pub data: [Float; 4]
}



impl Display for Matrix2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}] \n [{}, {}] \n",
            self[0], self[1],
            self[2], self[3]
        )
    }
}



impl Index<usize> for Matrix2 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Matrix2 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Matrix2 {

    pub fn new(data: [Float; 4]) -> Matrix2 {
        Matrix2 {
            data
        }
    }



    pub fn id() -> Matrix2 {
        Matrix2::new([
            1., 0.,
            0., 1.
        ])
    }



    pub fn transpose(&self) -> Matrix2 {
        Matrix2::new([
            self[0], self[2],
            self[1], self[3]
        ])
    }



    pub fn det(&self) -> Float {

        let a = self[0];
        let b = self[1];
        let c = self[2];
        let d = self[3];
        
        a * d - b * c
    }



    pub fn trace(&self) -> Float {

        self[0] + self[3]

    }



    pub fn eig(&self) -> Option<(f64, f64)> {

        let t = self.trace();

        let d = self.det();

        let r = t.powf(2.) - 4. * d;

        if r < 0. {
           return None;
        }

        let y1 = (t + r.sqrt()) / 2.;

        let y2 = (t - r.sqrt()) / 2.;

        Some((y1, y2))
    }



    pub fn inv(&self) -> Option<Matrix2> {

        let d = self.det();

        if d == 0. {
            return None;
        }

        let a = self[0];
        let b = self[1];
        let c = self[2];
        let d = self[3];

        let mut m = matrix2![
            d, -b,
           -c,  a
        ];

        m = m * (1. / d);

        Some(m)
    }



    pub fn rotation(rad: f64) -> Matrix2 {

        let z = matrix2![
            rad.cos(), -rad.sin(),
            rad.sin(),  rad.cos()
        ];

        z
    }



    pub fn scale(s: f64) -> Matrix2 {
        matrix2![
            s,  0.,
            0., s
        ]
    }



    pub fn sheer(xy:f64, yx:f64) -> Matrix2 {
        matrix2![
            1., yx,
            xy, 1.
        ]
    }


    
    pub fn from_basis(
        x: Vector2,
        y: Vector2
    ) -> Matrix2 {
        let r = compose_basis2![
            &x,
            &y
        ];

        r
    }
    


    pub fn into_basis(&self) -> [Vector2; 2] {

        let x = vec2![self[0], self[2]];
        let y = vec2![self[1], self[3]];

        [x,y]
    }
    


    pub fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
        self[2] = f(self[2]); 
        self[3] = f(self[3]);
    }
}



fn add(a:&Matrix2, b:&Matrix2) -> Matrix2 {
    Matrix2::new([
        a[0] + b[0], a[1] + b[1],
        a[2] + b[2], a[3] + b[3]
    ])
}



fn sub(a:&Matrix2, b:&Matrix2) -> Matrix2 {
    Matrix2::new([
        a[0] - b[0], a[1] - b[1],
        a[2] - b[2], a[3] - b[3]
    ])
}



fn mul(a:&Matrix2, b:&Matrix2) -> Matrix2 {
    Matrix2::new([
        a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[2],
        a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[2]
    ])
}



fn mul_v(a:&Matrix2, v:&Vector2) -> Vector2 {
    Vector2::new(
        a[0] * v.x + a[1] * v.y,
        a[3] * v.x + a[4] * v.y
    )
}



impl Add for &Matrix2 {
    type Output = Matrix2;

    fn add(self, b:&Matrix2) -> Matrix2 {
        add(self, b)
    }
}



impl Add for Matrix2 {
    type Output = Matrix2;

    fn add(self, b:Matrix2) -> Matrix2 {
        add(&self, &b)
    }
}



impl Sub for &Matrix2 {
    type Output = Matrix2;

    fn sub(self, b:&Matrix2) -> Matrix2 {
        sub(self, b)
    }
}



impl Sub for Matrix2 {
    type Output = Matrix2;

    fn sub(self, b:Matrix2) -> Matrix2 {
        sub(&self, &b)
    }
}



impl Mul for &Matrix2 {
    type Output = Matrix2;

    fn mul(self, b: &Matrix2) -> Matrix2 {
        mul(self, b)
    }
}



impl Mul for Matrix2 {
    type Output = Matrix2;

    fn mul(self, b: Matrix2) -> Matrix2 {
        mul(&self, &b)
    }
}



impl Mul <Vector2> for &Matrix2 {
    type Output = Vector2;

    fn mul(self, v:Vector2) -> Vector2 {
        mul_v(&self, &v)
    }
}



impl Mul <Vector2> for Matrix2 {
    type Output = Vector2;

    fn mul(self, v:Vector2) -> Vector2 {
        mul_v(&self, &v)
    }
}



impl Mul <&Vector2> for Matrix2 {
    type Output = Vector2;

    fn mul(self, v:&Vector2) -> Vector2 {
        mul_v(&self, v)
    }
}



impl Mul <&Vector2> for &Matrix2 {
    type Output = Vector2;

    fn mul(self, v:&Vector2) -> Vector2 {
        mul_v(self, v)
    }
}



impl Mul <Float> for Matrix2 {
    type Output = Matrix2;

    fn mul(mut self, s:f64) -> Matrix2 {
        self[0] *= s;
        self[1] *= s;
        self[2] *= s;
        self[3] *= s;
        self
    }
}



impl Div <Float> for Matrix2 {
    type Output = Matrix2;

    fn div(mut self, s:f64) -> Matrix2 {
        self[0] /= s;
        self[1] /= s;
        self[2] /= s;
        self[3] /= s;
        self
    }
}



impl AddAssign for Matrix2 {
    fn add_assign(&mut self, b:Matrix2) {
        self[0] += b[0];
        self[1] += b[1];
        self[2] += b[2];
        self[3] += b[3];
    }
}



impl SubAssign for Matrix2 {
    fn sub_assign(&mut self, b:Matrix2) {
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



impl PartialEq for Matrix2 {
    fn eq(&self, b: &Matrix2) -> bool {
        self[0] == b[0] &&
        self[1] == b[1] &&
        self[2] == b[2] &&
        self[3] == b[3]
    }
}



impl Eq for Matrix2 {}



impl Neg for Matrix2 {

    type Output = Matrix2;

    fn neg(mut self) -> Self {
        self[0] *= -1.;
        self[1] *= -1.;
        self[2] *= -1.;
        self[3] *= -1.;
        self
    }
}



impl From<Matrix<f64>> for Matrix2 {
    fn from(m: Matrix<f64>) -> Matrix2 {
        matrix2![
            m[[0,0]], m[[0,1]],
            m[[1,0]], m[[1,1]]
        ]
    }
}



impl From<&Matrix<f64>> for Matrix2 {
    fn from(m: &Matrix<f64>) -> Matrix2 {
        matrix2![
            m[[0,0]], m[[0,1]],
            m[[1,0]], m[[1,1]]
        ]
    }
}



impl From<Matrix4> for Matrix2 {
    fn from(m: Matrix4) -> Matrix2 {
        matrix2![
            m[0], m[1],
            m[4], m[5]
        ]
    }
}



impl From<&Matrix4> for Matrix2 {
    fn from(m: &Matrix4) -> Matrix2 {
        matrix2![
            m[0], m[1],
            m[4], m[5]
        ]
    }
}



mod tests {
    
    use super::{ Matrix, Vector, Matrix2 };
    
    #[test]
    fn eig_test() {

        let max = 6.3;
        let m = Matrix::rand(2, 2, max);
        let id = Matrix::id(2);
        let b = Vector::new(vec![0.,0.]);
        let m2: Matrix2 = m.clone().into();

        println!("\n m2 is {} det is {} \n", m2, m2.det());
        
        let e = m2.eig();

        if e.is_none() {
           return;
        }

        let e = e.unwrap();

        println!("\n A is {}, {} \n", m, m.rank());
        println!("\n e is {} {} \n", e.0, e.1);
        
        let A1 = &m - &(&id * e.0);
        let A2 = &m - &(&id * e.1);

        println!("\n A1 is {}, {} \n", A1, A1.rank());
        println!("\n A2 is {}, {} \n", A2, A2.rank());

        let lu_A1 = A1.lu();
        let x1 = A1.solve(&b, &lu_A1).unwrap();

        let lu_A2 = A2.lu();
        let x2 = A2.solve(&b, &lu_A2).unwrap();

        println!("\n x1 {} \n", x1);
        println!("\n x2 {} \n", x2);

        let y1: Vector<f64> = &A1 * &x1;
        let y2: Vector<f64> = &A1 * &x1;

        println!("\n y1 {} \n", y1);
        println!("\n y2 {} \n", y2);

        //rank
        //there is a vector in null space

        //assert!(false);
    }
}

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
use crate::{Float, matrix3, matrix3::Matrix3, vec4, vector4::Vector4, vector3::Vector3};
extern crate wasm_bindgen;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;
use super::matrix::Matrix;


#[macro_export]
macro_rules! matrix4 {
    (
        $x1:expr, $y1:expr, $z1:expr, $t1:expr,
        $x2:expr, $y2:expr, $z2:expr, $t2:expr,
        $x3:expr, $y3:expr, $z3:expr, $t3:expr,
        $x4:expr, $y4:expr, $z4:expr, $t4:expr
    ) => {
        {
            Matrix4::new([
                $x1, $y1, $z1, $t1,
                $x2, $y2, $z2, $t2,
                $x3, $y3, $z3, $t3,
                $x4, $y4, $z4, $t4
            ])
        }
    };
}



#[macro_export]
macro_rules! compose_basis4 {
    (
        $x:expr, 
        $y:expr, 
        $z:expr, 
        $t:expr
    ) => {
        {
            Matrix4::new([
                $x[0], $y[0], $z[0], $t[0],
                $x[1], $y[1], $z[1], $t[1],
                $x[2], $y[2], $z[2], $t[2],
                $x[3], $y[3], $z[3], $t[3]
            ])
        }
    };
}



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



#[macro_export]
macro_rules! compose_m4 {
    ($v:expr) => { $v };
    ($v:expr, $($x:expr),+) => {

        &( $v ) * &( compose_m4!($($x),*) )

    }
}



#[derive(Clone, Debug)]
pub struct Matrix4 {
    pub data: [Float; 16]
}



impl Display for Matrix4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, 
            "\n [{}, {}, {}, {}] \n [{}, {}, {}, {}] \n [{}, {}, {}, {}] \n [{}, {}, {}, {}] \n", 
            self[0], self[1], self[2], self[3],
            self[4], self[5], self[6], self[7],
            self[8], self[9], self[10], self[11],
            self[12], self[13], self[14], self[15]
        )
    }
}



impl Index<usize> for Matrix4 {
    type Output = Float;

    fn index(&self, idx:usize) -> &Float {
        &self.data[idx]
    }
}



impl IndexMut<usize> for Matrix4 {
    fn index_mut(&mut self, idx:usize) -> &mut Float {
        &mut self.data[idx]
    }
}



impl Index<[usize;2]> for Matrix4 {
    type Output = Float;

    fn index(&self, idx:[usize;2]) -> &Float {
        &self.data[idx[0] * 4 + idx[1]]
    }
}



impl IndexMut<[usize;2]> for Matrix4 {
    fn index_mut(&mut self, idx:[usize;2]) -> &mut Float {
        &mut self.data[idx[0] * 4 + idx[1]]
    }
}



impl Matrix4 {

    pub fn new(data: [Float; 16]) -> Matrix4 {
        Matrix4 {
            data
        }
    }



    pub fn id() -> Matrix4 {
        Matrix4::new([
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0., 
            0., 0., 0., 1. 
        ])
    }


    
    pub fn t(&mut self) {
        *self = Matrix4::new([
            self[0], self[4], self[8],  self[12],
            self[1], self[5], self[9],  self[13],
            self[2], self[6], self[10], self[14],
            self[3], self[7], self[11], self[15]
        ]);
    }



    pub fn det(&self) -> Float {
        let a = matrix3![
            self[5], self[6], self[7],
            self[9], self[10], self[11],
            self[13], self[14], self[15]
        ];
        let b = matrix3![
            self[4], self[6], self[7],
            self[8], self[10], self[11],
            self[12], self[14], self[15]
        ];
        let c = matrix3![
            self[4], self[5], self[7],
            self[8], self[9], self[11],
            self[12], self[13], self[15]
        ];
        let d = matrix3![
            self[4], self[5], self[6],
            self[8], self[9], self[10],
            self[12], self[13], self[14]
        ];
        
        self[0] * a.det() - self[1] * b.det() + self[2] * c.det() - self[3] * d.det()
    }



    pub fn inv(&self) -> Option<Matrix4> {
        let s = self.det();
        
        if s==0. {
            return None;
        }

        let x1 = matrix3![
            self[5],  self[6],  self[7],
            self[9],  self[10], self[11],
            self[13], self[14], self[15]
        ];
        let y1 = matrix3![
            self[4],  self[6],  self[7],
            self[8],  self[10], self[11],
            self[12], self[14], self[15]
        ];
        let z1 = matrix3![
            self[4],  self[5],  self[7],
            self[8],  self[9], self[11],
            self[12], self[13], self[15]
        ];
        let t1 = matrix3![
            self[4],  self[5],  self[6],
            self[8],  self[9],  self[10],
            self[12], self[13], self[14]
        ];

        let x2 = matrix3![
            self[1],  self[2],  self[3],
            self[9],  self[10], self[11],
            self[13], self[14], self[15]
        ];
        let y2 = matrix3![
            self[0],  self[2],  self[3],
            self[8],  self[10], self[11],
            self[12], self[14], self[15]
        ];
        let z2 = matrix3![
            self[0],  self[1],  self[3],
            self[8],  self[9], self[11],
            self[12], self[13], self[15]
        ];
        let t2 = matrix3![
            self[0],  self[1],  self[2],
            self[8],  self[9],  self[10],
            self[12], self[13], self[14]
        ];

        let x3 = matrix3![
            self[1],  self[2],  self[3],
            self[5],  self[6], self[7],
            self[13], self[14], self[15]
        ];
        let y3 = matrix3![
            self[0],  self[2],  self[3],
            self[4],  self[6], self[7],
            self[12], self[14], self[15]
        ];
        let z3 = matrix3![
            self[0],  self[1],  self[3],
            self[4],  self[5],  self[7],
            self[12], self[13], self[15]
        ];
        let t3 = matrix3![
            self[0],  self[1],  self[2],
            self[4],  self[5],  self[6],
            self[12], self[13], self[14]
        ];

        let x4 = matrix3![
            self[1],  self[2],  self[3],
            self[5],  self[6],  self[7],
            self[9],  self[10], self[11]
        ];
        let y4 = matrix3![
            self[0],  self[2],  self[3],
            self[4],  self[6],  self[7],
            self[8],  self[10], self[11]
        ];
        let z4 = matrix3![
            self[0],  self[1],  self[3],
            self[4],  self[5],  self[7],
            self[8],  self[9],  self[11]
        ];
        let t4 = matrix3![
            self[0],  self[1],  self[2],
            self[4],  self[5],  self[6],
            self[8],  self[9],  self[10]
        ];

        let mut m = matrix4![
            x1.det(), -y1.det(), z1.det(), -t1.det(),
            -x2.det(), y2.det(), -z2.det(), t2.det(),
            x3.det(), -y3.det(), z3.det(), -t3.det(),
            -x4.det(), y4.det(), -z4.det(), t4.det()
        ];

        m.t();

        m = m * (1./s);

        Some(m)
    }



    pub fn rot_x(x_rad: f64) -> Matrix4 {

        let x = matrix4![
            1., 0.,          0.,          0.,
            0., x_rad.cos(),-x_rad.sin(), 0.,
            0., x_rad.sin(), x_rad.cos(), 0.,
            0., 0.,          0.,          1.
        ];

        x
    }



    pub fn rot_y(y_rad: f64) -> Matrix4 {

        let y = matrix4![
            y_rad.cos(),  0., y_rad.sin(), 0.,
            0.,           1., 0.,          0.,
           -y_rad.sin(),  0., y_rad.cos(), 0.,
            0.,           0., 0.,          1.
        ];

        y
    }



    pub fn rot_z(z_rad: f64) -> Matrix4 {

        let z = matrix4![
            z_rad.cos(),-z_rad.sin(), 0., 0.,
            z_rad.sin(), z_rad.cos(), 0., 0.,
            0.,          0.,          1., 0.,
            0.,          0.,          0., 1.
        ];

        z
    }



    pub fn rotation(x_rad: f64, y_rad: f64, z_rad: f64) -> Matrix4 {

        let x = Matrix4::rot_x(x_rad);
        let y = Matrix4::rot_y(y_rad);
        let z = Matrix4::rot_z(z_rad);

        x * y * z
    }



    pub fn scale(s: f64) -> Matrix4 {
        matrix4![
            s,  0., 0., 0.,
            0., s,  0., 0.,
            0., 0., s,  0.,
            0., 0., 0., 1.
        ]
    }

    

    pub fn translate(x: f64, y: f64, z:f64, t:f64) -> Matrix4 {
        matrix4![
            1., 0., 0., x,
            0., 1., 0., y,
            0., 0., 1., z,
            0., 0., 0., t
        ]
    }



    pub fn sheer_x(y: f64, z: f64) -> Matrix4 {
        matrix4![
            1., 0., 0., 0.,
            y,  1., 0., 0.,
            z,  0., 1., 0.,
            0., 0., 0., 1.
        ]
    }



    pub fn sheer_y(x: f64, z: f64) -> Matrix4 {
        matrix4![
            1., x,  0., 0.,
            0., 1., 0., 0.,
            0., z,  1., 0.,
            0., 0., 0., 1.
        ]
    }



    pub fn sheer_z(x: f64, y: f64) -> Matrix4 {
        matrix4![
            1., 0., x,  0.,
            0., 1., y,  0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.
        ]
    }
    


    pub fn sheer(
        xy:f64, xz:f64,
        yx:f64, yz:f64,
        zx:f64, zy:f64
    ) -> Matrix4 {
        matrix4![
            1., yx, zx, 0.,
            xy, 1., zy, 0.,
            xz, yz, 1., 0.,
            0., 0., 0., 1.
        ]
    }



    pub fn from_basis(
        x: Vector4,
        y: Vector4,
        z: Vector4,
        t: Vector4
    ) -> Matrix4 {
        let r = compose_basis4![
            &x,
            &y,
            &z,
            &t
        ];

        r
    }



    pub fn into_basis(&self) -> [Vector4; 4] {
        let x = vec4![self[0], self[4], self[8],  self[12]];
        let y = vec4![self[1], self[5], self[9],  self[13]];
        let z = vec4![self[2], self[6], self[10], self[14]];
        let t = vec4![self[3], self[7], self[11], self[15]];

        [x,y,z,t]
    }


    
    pub fn orthographic(left: f64, right: f64, top: f64, bottom: f64, near: f64, far: f64) -> Matrix4 {
        let xx = 2.0 / ( right - left );
        let yy = 2.0 / ( top - bottom );
        let zz = -2.0 / ( far - near );
        
        let x = ( right + left ) * xx / 2.;
		let y = ( top + bottom ) * yy / 2.;
		let z = ( far + near ) * -zz / 2.;

        matrix4![
            xx, 0., 0., -x,
            0., yy, 0., -y,
            0., 0., zz, -z,
            0., 0., 0.,  1.
        ]
    }



    //what it does to a shape/space
    //how it diverges from identity
    //play with each parameter
    //what is the purpose of the matrix
    //action per component
    pub fn perspective(fov: f64, aspect: f64, near:f64, far:f64) -> Matrix4 {
        let f = (PI / 2. - fov / 2.).tan();
        let r = 1.0 / (near - far);

        let xx = f / aspect;
        let yy = f;
        let zz = (near + far) * r;
        let zt = near * far * r * 2.;
        let tz = -1.;

        matrix4![
            xx, 0., 0., 0.,
            0., yy, 0., 0.,
            0., 0., zz, tz,
            0., 0., zt, 0.
        ]
    }



    pub fn into_gl(&self) -> [f32; 16] {
        [
            self[0] as f32, self[1] as f32, self[2] as f32, self[3] as f32,
            self[4] as f32, self[5] as f32, self[6] as f32, self[7] as f32,
            self[8] as f32, self[9] as f32, self[10] as f32, self[11] as f32,
            self[12] as f32, self[13] as f32, self[14] as f32, self[15] as f32
        ]
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
        self[9]  = f(self[9]);
        self[10] = f(self[10]);
        self[11] = f(self[11]); 
        self[12] = f(self[12]);
        self[13] = f(self[13]);
        self[14] = f(self[14]);
        self[15] = f(self[15]);
    }
}



impl Add for &Matrix4 {
    type Output = Matrix4;

    fn add(self, b:&Matrix4) -> Matrix4 {
        Matrix4::new([
            self[0] + b[0], self[1] + b[1], self[2] + b[2], self[3] + b[3],
            self[4] + b[4], self[5] + b[5], self[6] + b[6], self[7] + b[7],
            self[8] + b[8], self[9] + b[9], self[10] + b[10], self[11] + b[11],
            self[12] + b[12], self[13] + b[13], self[14] + b[14], self[15] + b[15]
        ])
    }
}



impl AddAssign for Matrix4 {
    fn add_assign(&mut self, b:Matrix4) {
        self[0] += b[0];
        self[1] += b[1];
        self[2] += b[2];
        self[3] += b[3];
        self[4] += b[4];
        self[5] += b[5];
        self[6] += b[6];
        self[7] += b[7];
        self[8] += b[8];
        self[9]  += b[9];
        self[10] += b[10];
        self[11] += b[11]; 
        self[12] += b[12];
        self[13] += b[13];
        self[14] += b[14];
        self[15] += b[15];
    }
}



impl Sub for &Matrix4 {
    type Output = Matrix4;

    fn sub(self, b:&Matrix4) -> Matrix4 {
        Matrix4::new([
            self[0] - b[0], self[1] - b[1], self[2] - b[2], self[3] - b[3],
            self[4] - b[4], self[5] - b[5], self[6] - b[6], self[7] - b[7],
            self[8] - b[8], self[9] - b[9], self[10] - b[10], self[11] - b[11],
            self[12] - b[12], self[13] - b[13], self[14] - b[14], self[15] - b[15]
        ])
    }
}



impl SubAssign for Matrix4 {
    fn sub_assign(&mut self, b:Matrix4) {
        self[0] -= b[0];
        self[1] -= b[1];
        self[2] -= b[2];
        self[3] -= b[3];
        self[4] -= b[4];
        self[5] -= b[5];
        self[6] -= b[6];
        self[7] -= b[7];
        self[8] -= b[8];
        self[9]  -= b[9];
        self[10] -= b[10];
        self[11] -= b[11]; 
        self[12] -= b[12];
        self[13] -= b[13];
        self[14] -= b[14];
        self[15] -= b[15];
    }
}



impl Mul for Matrix4 {
    type Output = Matrix4;

    fn mul(self, b: Matrix4) -> Matrix4 {
        Matrix4::new([
            self[0] * b[0] + self[1] * b[4] + self[2] * b[8] + self[3] * b[12],
            self[0] * b[1] + self[1] * b[5] + self[2] * b[9] + self[3] * b[13], 
            self[0] * b[2] + self[1] * b[6] + self[2] * b[10] + self[3] * b[14], 
            self[0] * b[3] + self[1] * b[7] + self[2] * b[11] + self[3] * b[15], 

            self[4] * b[0] + self[5] * b[4] + self[6] * b[8] + self[7] * b[12],
            self[4] * b[1] + self[5] * b[5] + self[6] * b[9] + self[7] * b[13], 
            self[4] * b[2] + self[5] * b[6] + self[6] * b[10] + self[7] * b[14], 
            self[4] * b[3] + self[5] * b[7] + self[6] * b[11] + self[7] * b[15], 
            
            self[8] * b[0] + self[9] * b[4] + self[10] * b[8] + self[11] * b[12],
            self[8] * b[1] + self[9] * b[5] + self[10] * b[9] + self[11] * b[13], 
            self[8] * b[2] + self[9] * b[6] + self[10] * b[10] + self[11] * b[14], 
            self[8] * b[3] + self[9] * b[7] + self[10] * b[11] + self[11] * b[15], 

            self[12] * b[0] + self[13] * b[4] + self[14] * b[8] + self[15] * b[12],
            self[12] * b[1] + self[13] * b[5] + self[14] * b[9] + self[15] * b[13], 
            self[12] * b[2] + self[13] * b[6] + self[14] * b[10] + self[15] * b[14], 
            self[12] * b[3] + self[13] * b[7] + self[14] * b[11] + self[15] * b[15]
        ])
    }
}



impl Mul for &Matrix4 {
    type Output = Matrix4;

    fn mul(self, b: &Matrix4) -> Matrix4 {
        Matrix4::new([
            self[0] * b[0] + self[1] * b[4] + self[2] * b[8] + self[3] * b[12],
            self[0] * b[1] + self[1] * b[5] + self[2] * b[9] + self[3] * b[13], 
            self[0] * b[2] + self[1] * b[6] + self[2] * b[10] + self[3] * b[14], 
            self[0] * b[3] + self[1] * b[7] + self[2] * b[11] + self[3] * b[15], 

            self[4] * b[0] + self[5] * b[4] + self[6] * b[8] + self[7] * b[12],
            self[4] * b[1] + self[5] * b[5] + self[6] * b[9] + self[7] * b[13], 
            self[4] * b[2] + self[5] * b[6] + self[6] * b[10] + self[7] * b[14], 
            self[4] * b[3] + self[5] * b[7] + self[6] * b[11] + self[7] * b[15], 
            
            self[8] * b[0] + self[9] * b[4] + self[10] * b[8] + self[11] * b[12],
            self[8] * b[1] + self[9] * b[5] + self[10] * b[9] + self[11] * b[13], 
            self[8] * b[2] + self[9] * b[6] + self[10] * b[10] + self[11] * b[14], 
            self[8] * b[3] + self[9] * b[7] + self[10] * b[11] + self[11] * b[15], 

            self[12] * b[0] + self[13] * b[4] + self[14] * b[8] + self[15] * b[12],
            self[12] * b[1] + self[13] * b[5] + self[14] * b[9] + self[15] * b[13], 
            self[12] * b[2] + self[13] * b[6] + self[14] * b[10] + self[15] * b[14], 
            self[12] * b[3] + self[13] * b[7] + self[14] * b[11] + self[15] * b[15]
        ])
    }
}



impl Mul <Float> for Matrix4 {
    type Output = Matrix4;

    fn mul(mut self, s:f64) -> Matrix4 {
        self[0] *= s;
        self[1] *= s;
        self[2] *= s;
        self[3] *= s;
        self[4] *= s;
        self[5] *= s;
        self[6] *= s;
        self[7] *= s;
        self[8] *= s;
        self[9] *= s;
        self[10] *= s;
        self[11] *= s;
        self[12] *= s;
        self[13] *= s;
        self[14] *= s;
        self[15] *= s;
        self
    }
}



impl Mul <Vector4> for &Matrix4 {
    type Output = Vector4;

    fn mul(self, v:Vector4) -> Vector4 {
        vec4![
            self[0] * v[0] + self[1] * v[1] + self[2] * v[2] + self[3] * v[3],
            self[4] * v[0] + self[5] * v[1] + self[6] * v[2] + self[7] * v[3],
            self[8] * v[0] + self[9] * v[1] + self[10] * v[2] + self[11] * v[3],
            self[12] * v[0] + self[13] * v[1] + self[14] * v[2] + self[15] * v[3]
        ]
    }
}



impl Mul <Vector4> for Matrix4 {
    type Output = Vector4;

    fn mul(self, v:Vector4) -> Vector4 {
        vec4![
            self[0] * v[0] + self[1] * v[1] + self[2] * v[2] + self[3] * v[3],
            self[4] * v[0] + self[5] * v[1] + self[6] * v[2] + self[7] * v[3],
            self[8] * v[0] + self[9] * v[1] + self[10] * v[2] + self[11] * v[3],
            self[12] * v[0] + self[13] * v[1] + self[14] * v[2] + self[15] * v[3]
        ]
    }
}



impl Div <Float> for Matrix4 {
    type Output = Matrix4;
    
    fn div(mut self, s:f64) -> Matrix4 {
        self[0] /= s;
        self[1] /= s;
        self[2] /= s;
        self[3] /= s;
        self[4] /= s;
        self[5] /= s;
        self[6] /= s;
        self[7] /= s;
        self[8] /= s;
        self[9] /= s;
        self[10] /= s;
        self[11] /= s;
        self[12] /= s;
        self[13] /= s;
        self[14] /= s;
        self[15] /= s;
        self
    }
}



fn eq(a: &Matrix4, b: &Matrix4) -> bool {
    //f64::EPSILON
    a[0] == b[0] &&
    a[1] == b[1] &&
    a[2] == b[2] &&
    a[3] == b[3] &&
    a[4] == b[4] &&
    a[5] == b[5] &&
    a[6] == b[6] &&
    a[7] == b[7] &&
    a[8] == b[8] &&
    a[9] == b[8] &&
    a[10] == b[10] &&
    a[11] == b[11] &&
    a[12] == b[12] &&
    a[13] == b[13] &&
    a[14] == b[14] &&
    a[15] == b[15]
}



fn almost_eq(a: &Matrix4, b: &Matrix4) -> bool {
    
    fn eq(a: Float, b: Float) -> bool {
        (a - b).abs() < f32::EPSILON as f64
    }

    eq(a[0], b[0]) &&
    eq(a[1], b[1]) &&
    eq(a[2], b[2]) &&
    eq(a[3], b[3]) &&
    eq(a[4], b[4]) &&
    eq(a[5], b[5]) &&
    eq(a[6], b[6]) &&
    eq(a[7], b[7]) &&
    eq(a[8], b[8]) &&
    eq(a[9], b[9]) &&
    eq(a[10], b[10]) &&
    eq(a[11], b[11]) &&
    eq(a[12], b[12]) &&
    eq(a[13], b[13]) &&
    eq(a[14], b[14]) &&
    eq(a[15], b[15])
}



impl PartialEq for Matrix4 {
    fn eq(&self, b: &Matrix4) -> bool {
        //eq(self, b)
        almost_eq(self, b)
    }
}



impl Eq for Matrix4 {}



impl Neg for Matrix4 {

    type Output = Matrix4;
    
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
        self[9] *= -1.;
        self[10] *= -1.;
        self[11] *= -1.;
        self[12] *= -1.;
        self[13] *= -1.;
        self[14] *= -1.;
        self[15] *= -1.;
        self
    }
}



impl From<Matrix3> for Matrix4 {
    fn from(m: Matrix3) -> Matrix4 {
        matrix4![
            m[0], m[1], m[2], 0.,
            m[3], m[4], m[5], 0.,
            m[6], m[7], m[8], 0.,
            0.,   0.,   0.,   1.
        ]
    }
}



mod tests {
    use std::{f32::EPSILON, f64::consts::PI};
    use crate::{ vec3 };
    use super::{
        Vector3,
        Matrix3,
        Matrix4,
        Matrix
    };

    //TODO!
    #[test]
    fn gimbal_lock() {
        
        let basis = Matrix3::id().into_basis();
        let v = vec3![2., 3., 4.];

        let x_rad = PI / 10.;
        let y_rad = PI / 5.; //PI / 2.;
        let z_rad = PI / 10.;
        
        let r1: Matrix3 = Matrix4::rotation(x_rad, y_rad, z_rad).into();
        
        let x_rad2 = PI / 10.;
        let y_rad2 = PI / 10.;
        let z_rad2 = PI / 10.;

        let r2: Matrix3 = Matrix4::rotation(x_rad2, y_rad2, z_rad2).into();
        let r1v = &r1 * &v;
        let r2v = &r2 * &v;
        
        let r3: Matrix3 = Matrix4::rotation(x_rad + x_rad2, y_rad + y_rad2, z_rad + z_rad2).into();
        let r2r1: Matrix3 = &r2 * &r1;
        
        let r3v = &r3 * &v;
        let r2r1v = &r2r1 * &v;
        
        println!("\n r1 is {} \n r2 is {} \n r3 is {} \n r2r1 is {} \n", r1, r2, r3, r2r1);

        println!("\n r3v is {} \n r2r1v {} \n", r3v, r2r1v);

        let x = basis[0];
        let y = basis[1];
        let z = basis[2];

        let r3v_x = r3v.angle(&x);
        let r3v_y = r3v.angle(&y);
        let r3v_z = r3v.angle(&z);

        let r2r1v_x = r2r1v.angle(&x);
        let r2r1v_y = r2r1v.angle(&y);
        let r2r1v_z = r2r1v.angle(&z);

        println!("\n r3v_x is {} \n r3v_y is {} \n r3v_z is {} \n", r3v_x, r3v_y, r3v_z);

        println!("\n r2r1v_x is {} \n r2r1v_y is {} \n r2r1v_z is {} \n", r2r1v_x, r2r1v_y, r2r1v_z);
    }
    
    #[test]
    fn perspective() {
        
        const fov: f32 = 45. * std::f32::consts::PI / 180.;
        const far: f32 = 100.;
        const near: f32 = 0.1;
        const aspect: f32 = 1.; //-2.414213;
       
        //apply to cube
        let p = Matrix4::perspective(fov as f64, aspect as f64, near as f64, far as f64);
        
        //assert_eq!(1.,0., "perspective {}", p);
    }
}

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
use crate::{Float, vec3, vector3::Vector3};
use super::{vector::Vector, vector4::Vector4, matrix4::Matrix4, utils::{clamp, eq_eps_f64}};



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
        let a = self;
        let s = a.det();

        let a11 = a[0];
        let a12 = a[1];
        let a13 = a[2];

        let a21 = a[3];
        let a22 = a[4];
        let a23 = a[5];

        let a31 = a[6];
        let a32 = a[7];
        let a33 = a[8];

        if s==0. {
           return None;
        }
 
        let xx = a22 * a33 - a23 * a32;
        let xy = a12 * a33 - a13 * a32; 
        let xz = a12 * a23 - a13 * a22;

        let yx = a21 * a33 - a23 * a31;
        let yy = a11 * a33 - a13 * a31;
        let yz = a11 * a23 - a13 * a21; 

        let zx = a21 * a32 - a22 * a31;
        let zy = a11 * a32 - a12 * a31;
        let zz = a11 * a22 - a12 * a21;

        let mut m = matrix3![
            xx, -yx, zx,
           -xy,  yy,-zy,
            xz, -yz, zz
        ];
        
        m.t();

        m = m * (1./s);
        
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



    pub fn orthonormal(v:&Vector3) -> Matrix3 {
        let mut x = *v;
        
        let mut y = Vector3::rand(10.);
        
        while y * x == 0. {
            y = Vector3::rand(10.);
        }

        let p = y.project_on(&x);
        
        y -= p;

        let mut z = x.cross(&y);

        x.normalize();
        y.normalize();
        z.normalize();

        Matrix3::from_basis(x, y, z)
    }



    pub fn projection(A:&Matrix3, b:&Vector3) -> Vector3 {

        //TODO do not use inverse
        //solve AtA x = At b
        let mut At = A.clone();
        
        println!("\n projection: A {} \n", At);

        At.t();
        
        println!("\n projection: At {} \n", At);
        
        let aat: Matrix3 = &At * A;

        println!("\n projection: A * &At {} \n", aat);

        println!("\n projection: aat det {} \n", aat.det());
        
        let aati = aat.inv().unwrap();

        let id = &aat * &aati;

        let id2 = &aati * &aat;

        println!("\n projection: aati {} \n", aati);

        println!("\n projection: id {} \n", id);

        println!("\n projection: id2 {} \n", id2);

        let x = aati * (At * b);

        println!("\n projection: x {} \n", x);

        let r = A * x;

        println!("\n projection: r {} \n", r);

        r
    }



    pub fn solve() {
        //TODO
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
    use rand::Rng;
    use crate::core::{matrix4::Matrix4, vector4::Vector4};
    use super::{Matrix3, Vector3, eq_eps_f64};
    


    #[test]
    fn inv() {

        let a = matrix3![
            0.00000000046564561, 5., 44363463473474574.,
            0.00000000046346561, 3., 44574574574573.,
            0.0000000006461, 1., 7534534534534.
        ];

        let e = a.inv().unwrap();

        println!("\n e is {} \n", e);

        let id: Matrix3 = a * e;

        println!("\n id is {} \n", id);

        assert!(false);
    }



    //#[test]
    fn projection() {
        //TODO modify proj
        //make sure proj orth to cross of in space
        let max = 50.;
        let mut rng = rand::thread_rng();
        let id = Matrix3::id();
        let b = id.into_basis();
        let id_x = b[0];
        let x = Vector3::rand(10.);
        let y = Vector3::rand(10.);
        let c1 = rng.gen_range(-max,max);
        let c2 = rng.gen_range(-max,max);
        let z = (&x * c1) + (&y * c2);

        let m = Matrix3::from_basis(x,y,z);

        assert!(eq_eps_f64(m.det(), 0.), "matrix should be singular");

        let orth: Matrix3 = Matrix3::orthonormal(&x);
        let rot: Matrix3 = Matrix4::rotation(PI / 4., 0., 0.).into();
        let t: Vector3 = &(orth * rot) * y;
        let tp: Vector3 = Matrix3::projection(&m, &t);
        let d = tp.distance(&t);
        
        assert!(d != 0., "t,tp distance should not be zero");

        println!("\n t is {} | tp is {} \n", t, tp);
        
        let e: Vector3 = t - tp;
        let d: f64 = e * y;

        println!("\n e is {} | y is {} | d is {} \n", e, y, d);

        assert!(eq_eps_f64(d, 0.), "d should be zero {}", d);

        //projection on full space is the same (try id)
        //e is orthogonal
        //project cross get zero
        //apply M
        //x angle with y id
        //x angle with z id
        
        //rotate y around x
        //rotate y in id space 
        //apply rotation which took id x to x 
        
        //id x vs x
        //relative to y,z
        //original location pi/2 with y,z
        //current location
        
        let dst: Vector3 = x - id_x;
        //which rotation is going to produce this effect ???
        let r = Matrix4::rotation(dst.x, dst.y, dst.z);

        //A through rotational composition
        //B through construction of orthonormal basis
        //x - id x - decompose into rotations
        //what rotation i need per dim

        //take x - vector from plane
        //rotate y around x
        //rotation
        // 

        //vector inside subspace - invariant
        //x angle with z axis
        //rotate y rotation by this angle
    }
    


    #[test]
    fn orthonormal() {
        /*
        ORTHONORMAL

        Q * Qt = I

        Q is inverse of Qt verify!
        
        */
        //investigate composition of orthonormal basis with rotation
        //AB vs BA
        //does rotation action transferable into space A by applying rotation before/after ? 
        
        let v = Vector3::rand(1.);
        let m = Matrix3::orthonormal(&v);
        let basis = m.into_basis();

        let xl = basis[0].length();
        let yl = basis[1].length();
        let zl = basis[2].length();
        
        let a = v.angle(&basis[0]);
        assert!(eq_eps_f64(a, 0.), "angle with first vector should be zero {}", a);

        assert!(eq_eps_f64(xl, 1.), "xl length should be 1 {}", xl);
        assert!(eq_eps_f64(yl, 1.), "yl length should be 1 {}", yl);
        assert!(eq_eps_f64(zl, 1.), "zl length should be 1 {}", zl);

        let a: f64 = &basis[0] * &basis[1];
        let b: f64 = &basis[0] * &basis[2];
        assert!(eq_eps_f64(a, 0.), "a should be 0 {}", a);
        assert!(eq_eps_f64(b, 0.), "b should be 0 {}", b);
        
        let c: f64 = &basis[1] * &basis[0];
        let d: f64 = &basis[1] * &basis[2];
        assert!(eq_eps_f64(c, 0.), "c should be 0 {}", c);
        assert!(eq_eps_f64(d, 0.), "d should be 0 {}", d);

        let e: f64 = &basis[2] * &basis[1];
        let f: f64 = &basis[2] * &basis[0];
        assert!(eq_eps_f64(e, 0.), "e should be 0 {}", e);
        assert!(eq_eps_f64(f, 0.), "f should be 0 {}", f);
    }



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

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
use crate::{Float, vec2, vector2::Vector2};



#[macro_export]
macro_rules! matrix2 {
    ($x:expr, $y:expr, $z:expr, $t: expr) => {
        {
            Matrix2::new([$x,$y,$z,$t])
        }
    };
}



#[derive(Clone, Debug)]
pub struct Matrix2 {
    pub data: [Float; 4]
}

impl Display for Matrix2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n [{}, {}] \n [{}, {}] \n", self[0], self[1], self[2], self[3])
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

    fn new(data: [Float; 4]) -> Matrix2 {
        Matrix2 {
            data
        }
    }

    fn id() -> Matrix2 {
        Matrix2::new([
            1., 0.,
            0., 1.
        ])
    }
    
    fn t(mut self) -> Matrix2 {
        self = Matrix2::new([
            self[0], self[2],
            self[1], self[3]
        ]);
        self
    }

    fn inv(&self) -> Option<Matrix2> {
        let s = self.det();
        if s==0. {
            return None;
        }
        Some(
            Matrix2::new([
                self[3]/s, -self[1]/s,
               -self[2]/s,  self[0]/s
            ])
        )
    }

    fn det(&self) -> Float {
        (self[0] * self[3]) - (self[1] * self[2])
    }

    fn rot(r:Float) -> Matrix2 {
        Matrix2::new([
            r.cos(), -r.sin(),
            r.sin(),  r.cos(),
        ])
    }
    
    fn apply(&mut self, f: fn(f64) -> f64) {
        self[0] = f(self[0]);
        self[1] = f(self[1]);
        self[2] = f(self[2]); 
        self[3] = f(self[3]);
    }

    //Vec2[]
    fn get_columns() {}
    fn get_rows() {}

}



impl Add for &Matrix2 {
    type Output = Matrix2;

    fn add(self, b:&Matrix2) -> Matrix2 {
        Matrix2::new([
            self[0] + b[0], self[1] + b[1],
            self[2] + b[2], self[3] + b[3]
        ])
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



impl Sub for &Matrix2 {
    type Output = Matrix2;

    fn sub(self, b:&Matrix2) -> Matrix2 {
        Matrix2::new([
            self[0] - b[0], self[1] - b[1],
            self[2] - b[2], self[3] - b[3]
        ])
    }
}



impl SubAssign for Matrix2 {
    fn sub_assign(&mut self, b:Matrix2) {
        self[0] -= b[0];
        self[1] -= b[1];
        self[2] -= b[2]; 
        self[3] -= b[3];
    }
}



impl Mul for &Matrix2 {
    type Output = Matrix2;

    fn mul(self, b: &Matrix2) -> Matrix2 {
        Matrix2::new([
            self[0] * b[0] + self[1] * b[2], self[0] * b[1] + self[1] * b[3],
            self[2] * b[0] + self[3] * b[2], self[2] * b[1] + self[3] * b[3]
        ])
    }
}



impl MulAssign for Matrix2 {
    fn mul_assign(&mut self, b:Self) {
        *self = Matrix2::new([
            self[0] * b[0] + self[1] * b[2], self[0] * b[1] + self[1] * b[3],
            self[2] * b[0] + self[3] * b[2], self[2] * b[1] + self[3] * b[3]
        ])
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



impl Mul <Vector2> for Matrix2 {
    type Output = Vector2;

    fn mul(self, v:Vector2) -> Vector2 {
        vec2![ 
            self[0] * v[0] + self[1] * v[1],
            self[2] * v[0] + self[3] * v[1]
        ]
    }
}



impl Div <Float> for Matrix2 {
    type Output = Matrix2;
    
    fn div(mut self, s:f64) -> Matrix2 {
        if s==0. {
            return self;
        }
        self[0] /= s;
        self[1] /= s;
        self[2] /= s; 
        self[3] /= s;
        self
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



mod tests {
    use std::f64::consts::PI;

    use super::{
        Matrix2 
    };
    
    #[test]
    fn operations() {
        
        let a = Matrix2::new([1., 2., 3., 4.]);

        let b = a.clone();
        
        let mut r1 = &a + &b;
        
        assert_eq!(r1, matrix2![2., 4., 6., 8.], "add");

        r1[1] = 0.;

        assert_eq!(r1[1], 0., "index");

        let mut a1 = Matrix2::new([1., 2., 3., 4.]);

        let b1 = a.clone();

        a1 += b1;

        assert_eq!(a1, matrix2![2., 4., 6., 8.], "add_assign");
        
        let mut a2 = Matrix2::new([1., 2., 3., 4.]);

        let b2 = Matrix2::id();

        let g = &a2 - &b2;

        assert_eq!(g, matrix2![0., 2., 3., 3.], "sub");

        a2 -= b2;

        assert_eq!(a2, matrix2![0., 2., 3., 3.], "sub_assign");

        let a2 = Matrix2::new([1., 2., 3., 4.]);

        let k: Matrix2 = a2 * 2.;

        assert_eq!(k, matrix2![2., 4., 6., 8.], "scale");

        let mut p: Matrix2 = k / 2.;

        assert_eq!(p, matrix2![1., 2., 3., 4.], "div");
        
        p = -p;

        assert_eq!(p, matrix2![-1., -2., -3., -4.], "neg");

        assert_eq!(true, p == matrix2![-1., -2., -3., -4.], "eq");

        let id = Matrix2::id();

        assert_eq!(id, matrix2![1., 0., 0., 1.], "id");

        let t = a.clone().t();

        let mul = &t * &a;

        assert_eq!(mul[1], mul[2], "mul");

        let i = a.inv().unwrap();

        let mul2 = &a * &i;

        assert_eq!(mul2, id, "inv");

        assert_eq!(id.det(), 1., "det");
        
        let mut r = Matrix2::rot(PI);

        let mut r2 = Matrix2::rot(PI/2.);
        
        fn round(n:f64) -> f64 {
            (n as i32) as f64
        }

        r.apply(round);

        r2.apply(round);

        assert_eq!(matrix2![0., -1., 1., 0.], r2, "rot {}", r2);

        assert_eq!(matrix2![-1., 0., 0., -1.], r, "rot {}", r);
        
    }
}

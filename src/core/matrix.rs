#![allow(dead_code, warnings)]
#![feature(layout_for_ptr)]
//http://www.hpcavf.uclan.ac.uk/softwaredoc/sgi_scsl_html/sgi_html/ch03.html#ag5Plchri
use std::{any::TypeId, cell::RefCell, f32::EPSILON, future::Future, mem::*, pin::Pin, rc::Rc, task::{Context, Poll}, time::Instant};
use std::{
    collections::HashMap, 
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
        DivAssign,
        Neg
    }
};
extern crate wasm_bindgen;
extern crate num_cpus;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, workers::Workers};
use super::{matrix3::Matrix3, matrix4::Matrix4, multiply::{ multiply_threads, mul_blocks }, utils::{division_level, from_data_square, pack_mul_task, transfer_into_sab}};

/*
TODO 
workers factory
workers bounded by hardware concurrency 
reuse workers
spread available work through workers, establish queue
*/

//mul
//mul matrix vector
//rref
//lu
//solve
//det
//eig
//compose
//qr
//cholesky
//svd
/*
use Newton method
b known
solve -> Ax = b
compute Ax - using obtained x
compute difference d = Ax - b
transform this difference into vector in column space using:
Au = d - solve for u
next x = x - u (shifting x towards better accuracy)
*/



//jacobian
//conv
//house
//givens
//pooling (pick number per block defined by stride length)
//pickens
//fft
//rank
//stochastic
//markov
//controllability matrix
//diag
//identity
//perspective
//rotation
//reflection
//zeros
//permutation
//upshift_permutation
//downshift_permutation
//ones
//toeplitz
//symplectic
//hamiltonian
//krylov
//markov
//hamming
//graph
//dyad
//translation
//least squares
//projection
//min
//max
//avg
//trace
//add
//subtract
//mul float
//mul matrix
//transpose
//apply
//pow
//kronecker
//inv (pseudo, left, right)
//solve
//distance between two subspaces
//dist
//lui
//ref
//rref

//is_positive_definite
//is_invertible
//is_upshift_permutation
//is_downshift_permutation
//is_exchange_permutation
//is_identity
//is_upper_triangular
//is triangular ??? cut half! especially for Ax (where it matters for solution)
//is_lower_triangular
//is_diagonal
//is_banded
//is_square
//is_skew_hermitian
//is_hermitian
//is_skew_symmetric
//is_tridiagonal
//is_upper_bidiagonal
//is_lower_bidiagonal
//is_permutation
//is_upper_hessenberg
//is_lower_hessenberg

//hash table with entries tuples for indices
//assemble matrix from row vectors
//assemble matrix from column vectors
//matrix columns into vectors
//matrix rows into vectors
//multiply in threads A * col i - cols of B / N threads - k col per threads - assemble



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



#[macro_export]
macro_rules! matrix {
    (
        $t:ty,
        $($($x:expr),*);*
    ) => {
        {
            let mut rows: usize = 0;
            let mut c: usize = 0;
            let mut v: Vec<$t> = Vec::new();
            $(
                $(
                    c += 1;
                    v.push($x);
                )*
                rows += 1;
            )*
            rows -= 1;
            
            if rows == 0 {
                rows = 1;
            }

            c /= rows;
            let mut m : Matrix<$t> = Matrix::new(rows, c);
            m.from_vec(v);
            m
        }
    }
}



#[derive(Clone, Debug)]
pub struct Matrix <T> {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<T>
}



impl <T: Number> Matrix<T> {

    pub fn new(rows: usize, columns:usize) -> Matrix<T> {

        assert!(rows >= 1, "rows less equal zero");

        assert!(columns >= 1, "columns less equal zero");

        let c = T::from_i32(0).unwrap();

        let data = vec![c; rows * columns];
        
        Matrix {
            rows,
            columns,
            data
        }
    }
    


    pub fn transpose(&self) -> Matrix<T> {

        let mut t = Matrix::new(self.columns, self.rows);
    
        for i in 0..self.rows {
            for j in 0..self.columns {
                t[[j,i]] = self[[i,j]];
            }
        }
    
        t
    }



    pub fn from_vec(&mut self, v:Vec<T>) {

        assert_eq!(v.len(), self.rows * self.columns, "from_vec - incorrect size");

        self.data = v;
    }



    pub fn test(&mut self) {

        let first = self.data[0];

        first.to_ne_bytes();
    }



    pub fn into_sab(&mut self) -> js_sys::SharedArrayBuffer {

        let size = size_of::<T>();

        let len = self.size() * size;

        let mem = js_sys::SharedArrayBuffer::new(len as u32);

        let mut m = js_sys::Uint8Array::new( &mem );

        for(i,v) in self.data.iter().enumerate() {
            let next = v.to_ne_bytes();
            for j in 0..size {
                m.set_index((i * size + j) as u32, next[j]);
            }
        }

        mem
    }



    pub fn copy_to_f64(m: &Matrix<f64>, dst: &mut Float64Array) {

        for i in 0..m.rows {
            for j in 0..m.columns {
                let idx = i * m.columns + j;
                dst.set_index(idx as u32, m[[i,j]]);
            }
        }
    }



    pub fn copy_to_f32(m: &Matrix<f32>, dst: &mut Float32Array) {

        for i in 0..m.rows {
            for j in 0..m.columns {
                let idx = i * m.columns + j;
                dst.set_index(idx as u32, m[[i,j]]);
            }
        }
    }


    
    pub fn from_sab_f64(rows: usize, columns: usize, data: &SharedArrayBuffer) -> Matrix<f64> {

        let mut m = Matrix::new(rows, columns);

        let d = Float64Array::new(data);
        
        let size = rows * columns;

        let mut v = vec![0.; size];

        for i in 0..size {
            v[i] = d.get_index(i as u32);
        }

        m.data = v;
        

        m
    }
    


    pub fn depth(&self) -> i32 {
        
        assert!(self.rows==self.columns, "depth: matrix is not square");
        
        if self.rows < 4 {
            return 0;
        }

        let size = (self.rows * self.columns) as f64;

        let p = size.log(4.);

        assert!(p.fract()==0., "depth: matrix is not exact power of 4");
        
        p as i32
    }



    pub fn sum(&self) -> T {

        let mut acc = identities::zero();
    
        for i in 0..self.data.len() {
            acc = acc + self.data[i];
        }
        
        acc
    }



    pub fn strassen(&self, B:&Matrix<T>) -> Matrix<T> {
    
        let (
            A11, 
            A12, 
            A21, 
            A22
        ) = Matrix::decompose_blocks(self);
    
        let (
            B11, 
            B12, 
            B21, 
            B22
        ) = Matrix::decompose_blocks(B);
        
        let P1: Matrix<T> = &(&A11 + &A22) * &(&B11 + &B22); //TODO implement with consumption
        let P2: Matrix<T> = &(&A21 + &A22) * &B11;
        let P3: Matrix<T> = &A11 * &(&B12 - &B22);
        let P4: Matrix<T> = &A22 * &(&B21 - &B11);
        let P5: Matrix<T> = &(&A11 + &A12) * &B22;
        let P6: Matrix<T> = &(&A21 - &A11) * &(&B11 + &B12);
        let P7: Matrix<T> = &(&A12 - &A22) * &(&B21 + &B22);
    
        let C11: Matrix<T> = &(&(&P1 + &P4) - &P5) + &P7;
        let C12: Matrix<T> = &P3 + &P5;
        let C21: Matrix<T> = &P2 + &P4;
        let C22: Matrix<T> = &(&(&P1 + &P3) - &P2) + &P6;
        let C = Matrix::recombine_blocks(C11, C12, C21, C22);
        
        C
    }
    
    
    
    pub fn decompose_blocks(A: &Matrix<T>) -> (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>) {
    
        assert_eq!(A.rows, A.columns, "matrix should be square");
        assert_eq!((A.rows as f32).log2().fract(), 0., "matrix should be pow 2");
    
        let r = A.rows / 2;
        let c = A.columns / 2;
        
        let mut A11 = Matrix::new(r, c);
        let mut A12 = Matrix::new(r, c);
        let mut A21 = Matrix::new(r, c);
        let mut A22 = Matrix::new(r, c);
        
        for i in 0..r {
            for j in 0..c {
                A11[[i, j]] = A[[i, j]];
                A12[[i, j]] = A[[i, c + j]];
                A21[[i, j]] = A[[r + i, j]];
                A22[[i, j]] = A[[r + i, c + j]];
            }
        }
    
    
        (A11, A12, A21, A22)
    }
    
    
    
    pub fn recombine_blocks(A11: Matrix<T>, A12: Matrix<T>, A21: Matrix<T>, A22: Matrix<T>) -> Matrix<T> {
    
        assert_eq!(A11.rows, A11.columns, "A11 matrix should be square");
        assert_eq!((A11.rows as f32).log2().fract(), 0., "A11 matrix should be pow 2");
        
        assert_eq!(A11.rows, A12.rows, "A11 should have rows equivalent to A12");
        assert_eq!(A11.columns, A12.columns, "A11 should have columns equivalent to A12");
        
        assert_eq!(A11.rows, A21.rows, "A11 should have rows equivalent to A21");
        assert_eq!(A11.columns, A21.columns, "A11 should have columns equivalent to A21");
    
        assert_eq!(A11.rows, A22.rows, "A11 should have rows equivalent to A22");
        assert_eq!(A11.columns, A22.columns, "A11 should have columns equivalent to A22");
    
        let rows = A11.rows;
        let columns = A11.columns;
        let r = rows * 2;
        let c = columns * 2;
    
        let mut A = Matrix::new(r, c);
    
        for i in 0..rows {
            for j in 0..columns {
                A[[i,j]] = A11[[i,j]];
                A[[i,j + columns]] = A12[[i,j]];
                A[[i + rows,j]] = A21[[i,j]];
                A[[i + rows,j + columns]] = A22[[i,j]];
            }
        } 
    
        A
    }



    pub fn id(size: usize) -> Matrix<T> {

        let mut data: Vec<T> = Vec::with_capacity(size * size);
        
        for i in 0..size {
            data[(size * i) + i] = T::from_i32(1).unwrap();
        }
        
        Matrix {
            data,
            rows: size,
            columns: size
        }
    }



    fn lu (A: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
        //TODO edge cases - irregular matrices shapes
        //TODO rearrange rows to reduce numerical errors before computation begins, mangle indices, how and why ?
        //TODO handle pivot zero case - rows rearrangement - create indices map
    
        let size = A.rows * A.columns;
        
        let mut L: Matrix<T> = Matrix::id(A.rows);
        
        let mut U: Matrix<T> = A.clone();
        
        for i in 0..(U.rows - 1) {
            let p = U.data[((i * U.columns) + i) as usize]; 
    
            let mut tmp: Matrix<T> = Matrix::id(A.rows);
    
            for j in (i + 1)..U.rows {
                let e = U.data[((j * U.columns) + i) as usize];
                let c = T::from_i32(-1).unwrap() * (e/p);
                
                tmp.data[((j * U.columns) + i) as usize] = c; 
                
                for k in i..U.columns {
                    let idx1 = ((i * U.columns) + k) as usize;
                    let idx2 = ((j * U.columns) + k) as usize;
                    U.data[idx2] = U.data[idx2] + (c * U.data[idx1]);
                }
            }
            
            let m = Matrix {
                data: tmp.data,
                rows: U.rows,
                columns: U.columns
            };
            
            //println!("L {} {:?}", i, m);
    
            L = multiply(&m, &L); //TODO improve
        }
    
        (L,U)
    }

    

    pub fn rand_shape(max_side: usize, max:f64) -> Matrix<T> {
        
        let mut rng = rand::thread_rng();
        
        let rows = rng.gen_range(0, max_side) + 1; 

        let columns = rng.gen_range(0, max_side) + 1;

        Matrix::rand(rows, columns, max)
    }
    


    pub fn rand(rows: usize, columns: usize, max: f64) -> Matrix<T> {

        let mut A = Matrix::new(rows, columns);

        let mut rng = rand::thread_rng();

        for i in 0..columns {
            for j in 0..rows {
                let value: f64 = rng.gen_range(-max, max);
                A[[j,i]] = T::from_f64(value).unwrap();
            }
        }

        A
    }
    


    pub fn init_const(A: &mut Matrix<T>, c: f64) {
        for i in 0..A.columns {
            for j in 0..A.rows {
                A[[j,i]] = T::from_f64(c).unwrap();
            }
        }
    }



    pub fn augment_sq2n_size(&self) -> usize {

        let A = self;

        if A.rows==A.columns && (A.rows as f32).log2().fract() == 0. {
           return A.rows;
        }
    
        let mut side = std::cmp::max(A.rows, A.columns);
    
        let l: f64 = (side as f64).log2();
    
        if l.fract() != 0. {
            side = (2. as f64).powf(l.ceil()) as usize;
        }
        
        side
    }



    pub fn augment_sq2n(&self) -> Matrix<T> {

        let A = self;

        let side = A.augment_sq2n_size();

        if side == A.rows && side == A.columns {
           return A.clone();
        }
        
        let mut m: Matrix<T> = Matrix::new(side, side);
    
        m = &m + A;
    
        m
    }



    pub fn size(&self) -> usize {

        self.rows * self.columns

    } 


    
    pub fn mem_size(&self) -> usize {

        size_of::<T>() * self.rows * self.columns

    }



    pub fn is_symmetric(&self) -> bool {

        if self.rows != self.columns || self.rows <= 1 {
           return false; 
        }

        for i in 0..(self.rows - 1) {

            let start = i + 1;

            for j in start..self.columns {
                
                if self[[i,j]] != self[[j,i]] {

                    return false;

                }
            }
        }

        true
    }
}



fn add <T: Number>(A: &Matrix<T>, B: &Matrix<T>, dim_a:bool) -> Matrix<T> {
    
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");

    let rows = if dim_a { A.rows } else { B.rows };
    let columns = if dim_a { A.columns } else { B.columns };
    let mut C: Matrix<T> = Matrix::new(rows, columns);
    
    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i,j]] = A[[i,j]] + B[[i,j]];
        }
    }
    
    C
}



fn subtract<T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");

    let mut C: Matrix<T> = Matrix::new(A.rows, A.columns);
    
    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i,j]] = A[[i,j]] - B[[i,j]];
        }
    }
    
    C
}



fn scale <T:Number>(m: &mut Matrix<T>, n: T) -> &mut Matrix<T> {
    
    m.data = m.data.iter().map(|x:&T| *x*n).collect();

    m
}



fn eq_f64<T: Number>(a: &Matrix<T>, b: &Matrix<T>) -> bool {
    
    if a.rows != b.rows || a.columns != b.columns {
       return false;
    }

    for i in 0..a.rows {
        for j in 0..a.columns {
            let d = a[[i,j]] - b[[i,j]];
            let dd: f64 = T::to_f64(&d).unwrap();
            if (dd).abs() > EPSILON as f64 {
                return false;
            }
        } 
    }
    
    true
}



fn eq<T: Number>(a: &Matrix<T>, b: &Matrix<T>) -> bool {
    
    if a.rows != b.rows || a.columns != b.columns {
       return false;
    }

    for i in 0..a.rows {
        for j in 0..a.columns {
            if a[[i,j]] != b[[i,j]] {
                return false;
            }
        } 
    }
    
    true
}



fn mul <T:Number>(
    A: &impl Index<[usize;2], Output = T>, 
    B: &impl Index<[usize;2], Output = T>,
    A_rows: usize,
    A_columns: usize,
    B_columns: usize
) -> Matrix<T> {
    
    let mut C: Matrix<T> = Matrix::new(A_rows, B_columns);
    
    for i in 0..A_rows {

        for j in 0..B_columns {

            for k in 0..A_columns {

                C[[i,j]] += A[[i,k]] * B[[k,j]];
            }
        }
    }

    C
}



pub fn get_optimal_depth<T: Number> (A: &Matrix<T>, optimal_element_size: usize) -> usize {

    assert_eq!(A.rows, A.columns, "A should be square matrix");

    let p = (optimal_element_size as f64).log(4.).ceil();

    let optimal_element_size = (4. as f64).powf(p) as usize;

    let size = A.rows * A.columns;

    if size < 64 {
        return 0;
    }
    
    let chunks = size / optimal_element_size;

    if chunks < 2 {
        return 4;
    }
    
    (chunks as f64).log(4.).ceil() as usize
}



pub fn multiply <T:Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {

    assert_eq!(A.columns, B.rows, "matrices dimensions should be compatible A columns {} B rows {}", A.columns, B.rows);
    
    mul(A, B, A.rows, A.columns, B.columns)
}



impl <T:Number> PartialEq for Matrix<T> {
    fn eq(&self, b: &Matrix<T>) -> bool {
        eq(self, b) //eq_f64(self, b)
    }
}



impl <T:Number> Eq for Matrix<T> {}



impl <T> Index<[usize;2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx:[usize;2]) -> &T {
        &self.data[self.columns * idx[0] + idx[1]]
    }
}



impl <T> IndexMut<[usize;2]> for Matrix<T> {
    fn index_mut(&mut self, idx:[usize;2]) -> &mut T {
        &mut self.data[self.columns * idx[0] + idx[1]]
    }
}



impl <T: Number> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut acc = "".to_string();

        for i in 0..self.rows {
            acc = acc + "[";
            for j in 0..self.columns {
                let n = self[[i,j]];
                let s = T::to_string(&n);
                acc = acc + &s;
                acc = acc + ",";
            }
            acc = acc + "] \n";
        }

        write!(f, "\n {} \n", acc)
    }
}



impl <T: Number> Add for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, b:&Matrix<T>) -> Matrix<T> {
        add(&self, b, true)
    }
}



impl <T: Number> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, b:Matrix<T>) -> Matrix<T> {
        add(&self, &b, true)
    }
}



impl <T: Number> Sub for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, b:&Matrix<T>) -> Matrix<T> {
        subtract(self, b)
    }
}



impl <T: Number> Sub for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, b:Matrix<T>) -> Matrix<T> {
        subtract(&self, &b)
    }
}



impl <T: Number> Mul for &Matrix<T> 
    where T: Number 
{
    type Output = Matrix<T>;

    fn mul(mut self, other: &Matrix<T>) -> Matrix<T> {
        multiply(self,other)
    }
}



impl <T: Number> Mul for Matrix<T> 
    where T: Number 
{
    type Output = Matrix<T>;

    fn mul(mut self, other: Matrix<T>) -> Matrix<T> {
        multiply(&self,&other)
    }
}



impl <T: Number> From<Matrix3> for Matrix<T> {

    fn from(m: Matrix3) -> Matrix<T> {

        let v: Vec<f64> = m.data.into();

        from_data_square(&v, 3)
    }
}



impl <T: Number> From<Matrix4> for Matrix<T> {

    fn from(m: Matrix4) -> Matrix<T> {

        let v: Vec<f64> = m.data.into();

        from_data_square(&v, 4)
    }
}



#[wasm_bindgen]
pub async fn test_multiplication(hc: f64) {

    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let performance = window.performance().unwrap();
    let c = 100;
    let max = (2. as f64).powf(3.);

    //TODO try to compare with exactly the same algorithm in single thread

    //TODO print difference when not equal
    //TODO establish concrete limits when it is reasonable to spawn threads ?
    //TODO am i discarding zero blocks here ??? check
    for i in 10..c {
        let optimal_block_size = 30 * i;
        let max_side = 4 * i;
        let mut A: Matrix<f32> = Matrix::rand_shape(max_side, max);
        //let mut B: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let mut B: Matrix<f32> = Matrix::rand(A.columns, A.rows, max);
        
        unsafe {
            log(&format!("\n multiplying A({}, {}) B({},{}) \n", A.rows, A.columns, B.rows, B.columns));
        }

        let start = performance.now();
        //TODO profile, when this is advantageous ?
        let r: Matrix<f32> = multiply_threads(hc as usize, optimal_block_size, &A, &B).await;
        //mul_blocks(&mut A, &mut B, optimal_block_size, false, hc as usize); 
        
        let end = performance.now();
        
        unsafe {
            log(&format!("\n by blocks {} \n", end - start));
        }

        let start = performance.now();
        let r2: Matrix<f32> = &A * &B;
        let end = performance.now();

        unsafe {
            log(&format!("\n naive {} \n", end - start));
        }

        let mut r3 = Matrix::new(A.rows, B.columns);
        Matrix::init_const(&mut r3, 0.);
        let r: Matrix<f32> = add(&r, &r3, false);
        

        
        if !(r == r2) {
            let diff: Matrix<f32> = &r - &r2;
            let s = diff.sum(); 
            unsafe {
                log(&format!("\n not equal, sum {} \n | r ({},{}) | r2 ({},{}) | {} \n", s, r.rows, r.columns, r2.rows, r2.columns, diff));

                //log(&format!("\n in threads {} \n in local {} \n", r, r2));
            }
            break;
        }

        unsafe {
            //log(&format!("\n without threads {} \n", end - start));
        }
        
        //assert!(r == r2, "they should be equal {} \n \n {}", r, r2);

        unsafe {
            //log(&format!("\n final result is {} \n \n {} \n", r, r2));
        }
    }
}



mod tests {
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ core::matrix::{ Matrix }, matrix };
    use super::{ get_optimal_depth, eq_f64, multiply, mul_blocks };
    
    

    #[test]
    fn transpose() {
        
        let t: Matrix<i32> = matrix![i32,
            1,2,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,4,5;
            1,2,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,1,2;
            4,5,4,5;
        ];
        let c = t.clone();
        let t2 = c.transpose();
        let r = multiply(&t,&t2);
        
        assert_eq!(r.is_symmetric(), true, "product of transposed matrices should be symmetric {}", r);

        //TODO transpose orthonormal basis, inverse
        
    }



    #[test]
    fn is_symmetric() {
        
        let m1 = matrix![f64,
            3., 1., 1., 1.;
            1., 2., 1., 1.;
            1., 1., 2., 1.;
            1., 1., 1., 2.;
        ];

        assert!(m1.is_symmetric(), "m1 is symmetric");

        let m2 = matrix![f64,
            4., 7.;
            7., 2.;
        ];

        assert!(m2.is_symmetric(), "m2 is symmetric");

        let m3 = matrix![f64,
            4., 4.;
            7., 2.;
        ];

        assert_eq!(m3.is_symmetric(), false, "m3 is not symmetric");
        
        let m4 = matrix![f64,
            3., 1., 1., 1.;
            1., 2., 1., 1.;
            1., 1., 2., 1.;
            1., 1., 1., 2.;
            1., 1., 1., 2.;
        ];
        
        assert_eq!(m4.is_symmetric(), false, "m4 is not symmetric");

        let m5 = matrix![f64,
            3., 1., 1., 3.33, 1., 10.1;
            1., 2., 1., 1., 1., 1.;
            1., 1., 5., 12., 1., 1.;
            3.33, 1., 12., 2., 1., 1.;
            1., 1., 1., 1., 4., 1.;
            10.1, 1., 1., 1., 1., 8.;
        ];
        
        assert_eq!(m5.is_symmetric(), true, "m5 is symmetric");
    }



    #[test]
    fn multiply_test() {
        
        let discard_zero_blocks = true;
        let threads = 10;

        for h in (1..2) {
            let max_side = 156;
            let max = 20000.;
            let optimal_block_size = 5000;
            let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
            let mut B: Matrix<f64> = Matrix::rand(A.columns, A.rows, max); //_shape(max_side, max);
        
            let C1: Matrix<f64> = &A * &B;
            let C: Matrix<f64> = mul_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, threads);
    
            assert_eq!(C.sum(), C1.sum(), "sum should be equal");
    
            for i in 0..C1.rows {
                for j in 0..C1.columns {
                    assert_eq!(C1[[i,j]], C1[[i,j]], "all entries should be equal");
                }
            }
        } 
    }   
    


    #[test]
    pub fn decomposition() {

        let t: Matrix<i32> = matrix![i32,
            3,3,1,2;
            3,3,1,2;
            4,4,5,2;
            4,4,5,2;
        ];
    
        let (A11, A12, A21, A22) = Matrix::decompose_blocks(&t);
        let s = format!("\n result \n {:?}, \n {:?}, \n {:?}, \n {:?}, \n", A11, A12, A21, A22);
        println!("{}", s);
    }



    #[test]
    fn augment_sq2n_test() {
        let iterations = 20;
        let max_side = 183;
        let max = 33333.;

        for i in 0..iterations {
            let m: Matrix<f64> = Matrix::rand_shape(max_side, max);
            let aug = Matrix::augment_sq2n(&m);
            let d = get_optimal_depth(&aug, 100);
            let size = (aug.rows * aug.columns) as f64;
    
            println!("({},{}) | size - {} | level - {}", aug.rows, aug.columns, size, d);
    
            let l: f64 = size.log2();

            assert_eq!(aug.rows, aug.columns, "from ({} {}) to ({} {}) - should be square", m.rows, m.columns, aug.rows, aug.columns);
            assert_eq!(l.fract(), 0., "from ({} {}) to ({} {}) - should be power of 2", m.rows, m.columns, aug.rows, aug.columns);
        }
    }



    #[test]
    fn strassen() {
        
        let max_side = 50;
        let max = 10.;
        let a: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let b: Matrix<f64> = Matrix::rand_shape(max_side, max);

        let s1 = Matrix::augment_sq2n_size(&a);

        let s2 = Matrix::augment_sq2n_size(&b);

        let s = std::cmp::max(s1,s2);

        let mut A: Matrix<f64> = Matrix::new(s,s); 

        A = &A + &a;

        let mut B: Matrix<f64> = Matrix::new(s,s); 

        B = &B + &b;

        let expected: Matrix<f64> = &A * &B;

        let s = format!("\n [{}] expected \n {:?} \n", expected.sum(), expected);

        println!("{}", s);
        
        let C = Matrix::strassen(&A, &B);
        
        let s = format!("\n [{}] result \n {:?} \n", C.sum(), C);
        
        println!("{}", s);

        let equal = eq_f64(&expected, &C);

        assert!(equal, "should be equal");
    }
}

#![allow(dead_code, warnings)]
#![feature(layout_for_ptr)]
//http://www.hpcavf.uclan.ac.uk/softwaredoc/sgi_scsl_html/sgi_html/ch03.html#ag5Plchri
use std::{any::TypeId, cell::RefCell, cmp::min, f32::EPSILON, future::Future, mem::*, pin::Pin, rc::Rc, task::{Context, Poll}, time::Instant};
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
use js_sys::{Float32Array, Float64Array, SharedArrayBuffer, Uint32Array, Uint8Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, vector, workers::Workers};
use super::{lu::{lu, lu_v2}, matrix3::Matrix3, matrix4::Matrix4, multiply::{ multiply_threads, strassen, mul_blocks, get_optimal_depth, decompose_blocks }, utils::eq_eps_f64, vector::Vector};

/*
TODO 
queue
workers factory
workers tasks
workers bounded by hardware concurrency 
reuse workers
spread available work through workers, establish queue
*/

//mul
//mul matrix vector
//rref
//lu
//det
//eig
//compose
//qr
//cholesky
//svd
/*
use Newton method
b known
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
            m.set_vec(v);
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
    


    pub fn exchange_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.columns {
            let t = self[[i, k]];
            self[[i, k]] = self[[j, k]];
            self[[j, k]] = t;
        }
    }



    pub fn exchange_columns(&mut self, i: usize, j: usize) {
        for k in 0..self.rows {
            let t = self[[k, i]];
            self[[k, i]] = self[[k, j]];
            self[[k, j]] = t;
        }
    }



    pub fn extract_leading_principal_submatrix(&self, n: usize) -> Matrix<T> {
        let mut A = Matrix::id(n);

        for i in 0..n {
            for j in 0..n {
                A[[i,j]] = self[[i,j]];
            }
        }

        A
    }



    pub fn lu (&self) -> lu<T> {
        lu_v2(self)
    }



    fn solve(&self, b: &Vector<T>, lu: &lu<T>) -> Vector<T> {
        let zero = T::from_f64(0.).unwrap();
        let Pb: Vector<T> = &lu.P * b;
        let PL: Matrix<T> = &lu.P * &lu.L;
        let mut y = Vector::new(vec![zero; PL.columns]);
        let mut x = Vector::new(vec![zero; lu.U.columns]);

        //TODO
        if !lu.d.is_empty() {
            return x;
        }
        
        for i in 0..PL.rows {
            let mut acc = b[i];

            for j in 0..(i + 1) {
                let c = PL[[i, j]];
                
                if j == i {
                    y[i] = acc / c;
                } else {
                    acc -= c * y[j];
                }
            }
        }
        
        for i in (0..lu.U.rows).rev() {
            let mut acc = y[i];

            for j in (i..lu.U.rows).rev() {
                let c = lu.U[[i, j]];
                
                if j == i {
                    x[i] = acc / c;
                } else {
                    acc -= c * x[j];
                }
            }
        }
        
        x
    }



    //TODO improve, assert n!
    fn generate_permutations(size: usize) -> Vec<Matrix<f32>> {
        let m: Matrix<f32> = Matrix::id(size);
        let mut list: Vec<Matrix<f32>> = Vec::new();
        
        for i in 0..size {
            for j in 0..size {
                if i != j {
                    let mut k = Matrix::id(size);
                    k.exchange_rows(i, j);
                    list.push(k);
                } 
            }
        }

        let l = list.len();
        for i in 0..l {
            let A = &list[l - 1 - i];
            let P = &list[i];
            let p = A * P;
            list.push(p);
        }

        list
    }

    

    
    
    


    pub fn inv(&self) -> Matrix<T> {
        //if matrix diagonal 1/e
        self.clone()

    }



    pub fn into_sab(&mut self) -> SharedArrayBuffer {

        let size = size_of::<T>();

        let len = self.size() * size;

        let mem = SharedArrayBuffer::new(len as u32);

        let mut m = Uint8Array::new( &mem );

        for(i,v) in self.data.iter().enumerate() {
            let next = v.to_ne_bytes();
            for j in 0..size {
                m.set_index((i * size + j) as u32, next[j]);
            }
        }

        mem
    }



    pub fn transfer_into_sab(A:&Matrix<f64>, B:&Matrix<f64>) -> SharedArrayBuffer {
        let sa = A.mem_size();
        let sb = B.mem_size();
        let sc = A.rows * B.columns * size_of::<f32>();
        let size = sa + sb + sc;
    
        unsafe {
            log(&format!("\n ready to allocated {} bytes in sab \n", size));
        }
    
        let mut s = SharedArrayBuffer::new(size as u32);
        
        let v = Float64Array::new(&s);
        v.fill(0., 0, v.length());
    
        let mut v1 = Float64Array::new_with_byte_offset(&s, 0);
        let mut v2 = Float64Array::new_with_byte_offset(&s, sa as u32); //should it be + 1 ?
    
        Matrix::<f64>::copy_to_f64(&A, &mut v1);
        Matrix::<f64>::copy_to_f64(&B, &mut v2);
    
        unsafe {
            log(&format!("\n allocated {} bytes in sab \n", size));
        }
    
        s
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
    


    pub fn copy_to_f64(m: &Matrix<f64>, dst: &mut Float64Array) {

        for i in 0..m.rows {
            for j in 0..m.columns {
                let idx = i * m.columns + j;
                dst.set_index(idx as u32, m[[i,j]]);
            }
        }
    }



    pub fn id(size: usize) -> Matrix<T> {

        let zero = T::from_i32(0).unwrap();
        
        let mut data: Vec<T> = vec![zero; size * size];
        
        for i in 0..size {
            data[(size * i) + i] = T::from_i32(1).unwrap();
        }
        
        Matrix {
            data,
            rows: size,
            columns: size
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



    pub fn init_const(A: &mut Matrix<T>, c: f64) {
        for i in 0..A.columns {
            for j in 0..A.rows {
                A[[j,i]] = T::from_f64(c).unwrap();
            }
        }
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



    pub fn rand_shape(max_side: usize, max:f64) -> Matrix<T> {
        
        let mut rng = rand::thread_rng();
        
        let rows = rng.gen_range(0, max_side) + 1; 

        let columns = rng.gen_range(0, max_side) + 1;

        Matrix::rand(rows, columns, max)
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



    pub fn set_vec(&mut self, v:Vec<T>) {

        assert_eq!(v.len(), self.rows * self.columns, "from_vec - incorrect size");

        self.data = v;
    }



    pub fn sum(&self) -> T {

        let mut acc = identities::zero();
    
        for i in 0..self.data.len() {
            acc = acc + self.data[i];
        }
        
        acc
    }



    pub fn size(&self) -> usize {

        self.rows * self.columns

    } 


    
    pub fn mem_size(&self) -> usize {

        size_of::<T>() * self.rows * self.columns

    }



    pub fn is_diagonally_dominant(&self) -> bool {

        if self.rows != self.columns {
            return false;
        }

        /*
        a square matrix is said to be diagonally dominant if, for every row of the matrix, 
        the magnitude of the diagonal entry in a row is larger than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row
        */
        false
    } 



    pub fn is_symmetric_positive_definite() {

    }



    pub fn is_positive_definite() {

    }



    pub fn is_identity(&self) -> bool {

        let zero = T::from_f64(0.).unwrap();
        
        let one = T::from_f64(1.).unwrap();

        for i in 0..self.rows {
            for j in 0..self.columns {
                if i == j {
                    if self[[i, j]] != one {
                        return false;
                    }
                } else {
                    if self[[i, j]] != zero {
                        return false;
                    }
                }
            }
        }

        true
    }



    pub fn is_diag(&self) -> bool {
        let zero = T::from_f64(0.).unwrap();

        for i in 0..self.rows {
            for j in 0..self.columns {
                if i == j {
                    continue;
                }
                if self[[i, j]] != zero {
                    return false;
                } 
            }
        }

        true
    }



    pub fn is_upper_triangular(&self) -> bool {
        
        let zero = T::from_f64(0.).unwrap();

        for i in 0..self.rows {

            for j in 0..i {

                if self[[i, j]] != zero {

                   return false;

                }
            }
        }

        true
    }



    pub fn is_lower_triangular(&self) -> bool {

        let zero = T::from_f64(0.).unwrap();

        for i in 0..(self.rows - 1) {

            for j in (i + 1)..self.columns {

                if self[[i, j]] != zero {

                    return false;

                }
            }
        }

        true
    }



    pub fn is_permutation(&self) -> bool {
        
        let one = T::from_i32(1).unwrap();
        
        let t = Vector::new(vec![one; self.columns]);
        
        let r = self * &t;

        r == t
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



fn eq_bound_eps<T: Number>(a: &Matrix<T>, b: &Matrix<T>) -> bool {
    
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



pub fn mul_v <T:Number>(A: &Matrix<T>, b: &Vector<T>) -> Vector<T> {

    assert_eq!(A.columns, b.data.len(), "matrix and vector dim incompatible A columns {} b len {}", A.columns, b.data.len());
    let z = T::from_f64(0.).unwrap();
    let d = vec![z; A.rows];
    let mut out = Vector::new(d);

    for i in 0..A.rows {
        for j in 0..A.columns {
            out[i] += A[[i, j]] * b[j];
        }
    }

    out
}



/*
TODO
mul_diag

if A.is_diag() {
        
    let mut C = B.clone();
    //TODO test
    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i, j]] *= A[[i, i]];        
        }
    }

    C
} else {

    mul(A, B, A.rows, A.columns, B.columns)
}
*/



pub fn multiply <T:Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {

    assert_eq!(A.columns, B.rows, "matrices dimensions should be compatible A columns {} B rows {}", A.columns, B.rows);
    
    mul(A, B, A.rows, A.columns, B.columns)
}



impl <T:Number> PartialEq for Matrix<T> {
    fn eq(&self, b: &Matrix<T>) -> bool {
        eq(self, b)
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
                if j < self.columns - 1 {
                    acc = acc + ", ";
                }
            }
            acc = acc + "]\n";
        }

        write!(f, "\n{}\n", acc)
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



impl <T: Number> Mul <&Vector<T>> for &Matrix<T> 
    where T: Number 
{
    type Output = Vector<T>;

    fn mul(mut self, b: &Vector<T>) -> Vector<T> {
        mul_v(self, b)
    }
}



impl <T: Number> Mul <Vector<T>> for Matrix<T> 
    where T: Number 
{
    type Output = Vector<T>;

    fn mul(mut self, b: Vector<T>) -> Vector<T> {
        mul_v(&self, &b)
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

        let d: Vec<f64> = m.data.into();

        let data: Vec<T> = d.iter().map(|x| { T::from_f64(*x).unwrap() }).collect();

        let mut m = Matrix::new(3,3);
    
        m.set_vec(data);
    
        m
    }
}



impl <T: Number> From<Matrix4> for Matrix<T> {

    fn from(m: Matrix4) -> Matrix<T> {

        let d: Vec<f64> = m.data.into();

        let data: Vec<T> = d.iter().map(|x| { T::from_f64(*x).unwrap() }).collect();

        let mut m = Matrix::new(4,4);
    
        m.set_vec(data);
    
        m
    }
}



pub struct P_compact <T> {
    pub map: Vec<T>
}



impl <T: Number> P_compact <T> {
    
    pub fn new(s: usize) -> P_compact<T> {
        let z = T::from_i32(0).unwrap();
        let mut map = vec![z; s];

        for i in 0..s {
            map[i] = T::from_i32(i as i32).unwrap();
        }

        P_compact {
            map
        }
    }



    pub fn exchange_rows(&mut self, i: usize, j: usize) {
        let r = self.map[i];
        let l = self.map[j];
        self.map[j] = r;
        self.map[i] = l;
    }



    pub fn exchange_columns(&mut self, i: usize, j: usize) {
        let r = T::to_i32(&self.map[i]).unwrap() as usize;
        let l = T::to_i32(&self.map[j]).unwrap() as usize;
        self.exchange_rows(r, l);
    }



    pub fn into_p(&self) -> Matrix<T> {
        let s = self.map.len();
        let mut m: Matrix<T> = Matrix::new(s, s);

        for i in 0..s {
            let j = T::to_i32(&self.map[i]).unwrap() as usize;
            m[[i, j]] = T::from_i32(1).unwrap();
        }

        m
    }



    pub fn into_p_t(&self) -> Matrix<T> {
        let s = self.map.len();
        let mut m: Matrix<T> = Matrix::new(s, s);

        for i in 0..s {
            let j = T::to_i32(&self.map[i]).unwrap() as usize;
            m[[j, i]] = T::from_i32(1).unwrap();
        }

        m
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
        let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
        //let mut B: Matrix<f64> = Matrix::rand_shape(max_side, max);
        let mut B: Matrix<f64> = Matrix::rand(A.columns, A.rows, max);
        
        unsafe {
            log(&format!("\n multiplying A({}, {}) B({},{}) \n", A.rows, A.columns, B.rows, B.columns));
        }

        let start = performance.now();
        //TODO profile, when this is advantageous ?
        let r: Matrix<f64> = multiply_threads(hc as usize, optimal_block_size, &A, &B).await;
        //mul_blocks(&mut A, &mut B, optimal_block_size, false, hc as usize); 
        
        let end = performance.now();
        
        unsafe {
            log(&format!("\n by blocks {} \n", end - start));
        }

        let start = performance.now();
        let r2: Matrix<f64> = &A * &B;
        let end = performance.now();

        unsafe {
            log(&format!("\n naive {} \n", end - start));
        }

        let mut r3 = Matrix::new(A.rows, B.columns);
        Matrix::init_const(&mut r3, 0.);
        let r: Matrix<f64> = add(&r, &r3, false);
        
        if !(r == r2) {
            let diff: Matrix<f64> = &r - &r2;
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




//TODO vector transpose into matrix, vector into matrix, matrix 1,n into vector, vector transpose

mod tests {
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ vector, core::matrix::{ Matrix }, matrix };
    use super::{ eq_eps_f64, Vector, P_compact, Number, get_optimal_depth, eq_bound_eps, multiply, mul_blocks, strassen, decompose_blocks };

    /*
    
    
    
    let rows = 5;

    let columns = 5;

    let max = 100.;

    let mut A: Matrix<f64> = Matrix::rand(rows, columns, max);

    let mut lu = A.lu();

    let R: Matrix<f64> = &lu.L * &lu.U;

    let PL: Matrix<f64> = &lu.P * &lu.L;
    */
    
    //rows > columns
    //columns > rows
    //zero pivots initially
    //edge cases - [1] [1,2,3] [1,2,3]T [0,0,0], all ones etc



    fn lu_test4() {

        //solve with free variables
        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 1., 1., 1.;
            1., 2.5, 2., 2., 12.;
            1., 2.9, 3., 1., 7.;
            1., 4., 4., 2., 3.; 
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(false);

    }



    fn lu_test3() {
        /*
        let mut A: Matrix<f32> = matrix![f32,
            1.3968362, -0.97569525, -5.018955, 0.7136311;
            -4.6254315, 9.305554, 2.5439813, 1.9787005;
            -3.233496, -4.881222, -3.2327516, 3.0223584;
            -1.1067164, -4.347563, -8.04766, 1.6895233;
        ];
        */
        let mut A: Matrix<f64> = matrix![f64,
            1.3968362, -0.0009525, -5.018955, 23352352350.7136311; //-0.00009525 TODO
            -4.6254315, 9.305554, 2.5439813, 1234234234.9787005;
            -3.233496, -4.881222, -3.2327516, 3534534534.0223584;
            -1.1067164, -445645645645.347563, -8.04766, 1634634634.6895233;
        ];
    
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test3 A should be equal to R \n");
    }



    fn lu_test2() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 1., 1.;
            1., 2., 2., 2.;
            1., 2., 3., 1.;
            1., 2., 4., 2.;
        ];
        
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test2 A should be equal to R \n");

        assert_eq!(lu.d.len(), 1, "\n lu_test2 d should contain 1 element \n");

        assert_eq!(lu.d[0], 1, "\n lu_test2 d should contain second col \n");
        
        assert!(id != lu.P, "\n lu_test2 P should not be identity \n");
    }



    fn lu_test1() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];
        
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert_eq!(A, R, "\n lu_test1 A should be equal to R \n");

        assert!(id != lu.P, "\n lu_test1 P should not be identity \n");
    }
    


    #[test]
    fn lu_test() {
        
        //lu_test1();

        //lu_test2();

        //lu_test3();

        lu_test4();
    }



    //#[test]
    fn solve_test() {
            
        let test = 10;

        for i in 1..test {
            println!("\n solve: working with {} \n", i);

            let size = i;
            let max = 10.;
            let A: Matrix<f64> = Matrix::rand(size, size, max);
            let b: Vector<f64> = Vector::rand(size as u32, max);
            let lu = A.lu();
            let x = A.solve(&b, &lu);
            let Ax = &A * &x;
            let PAx = &lu.P * &Ax;
            
            for j in 0..PAx.data.len() {
                let eq = eq_eps_f64(PAx[j], b[j]);
                if !eq {
                    println!("\n A is {} \n", A);

                    println!("\n x is {} \n", x);
            
                    println!("\n b is {} \n", b);
            
                    println!("\n Ax is {} \n", Ax);
            
                    println!("\n PAx is {} \n", PAx);
                }
                assert!(eq, "entries should be equal");
            }  
        }
    }



    #[test]
    fn vector_product() {

        let b = vector![1., 2., 3.]; 
        
        let A: Matrix<f32> = matrix![f32,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let Ab = A * b;
        
        assert_eq!(Ab[0], 14., "1 entry is 14");
        assert_eq!(Ab[1], 31., "2 entry is 31");
        assert_eq!(Ab[2], 22., "3 entry is 22");
    }



    #[test] 
    fn p_compact() {
        
        let mut p: P_compact<i32> = P_compact::new(4);

        //p.exchange_rows(1, 3);
        //p.exchange_rows(1, 2);
        
        p.exchange_columns(0, 3);

        let p_m = p.into_p();

        let p_m_t = p.into_p_t();

        println!("\n p is {} \n", p_m);

        println!("\n p t is {} \n", p_m_t);

        //assert!(false);
    }



    #[test] 
    fn perm_test() {
        let p = Matrix::<f32>::generate_permutations(4);

        for i in 0..p.len() {
            //println!("P next is {} \n", p[i]);
        }

        //assert!(false);
    }
    
    

    #[test]
    fn exchange_rows() {

        let mut m: Matrix<f32> = matrix![f32,
            -3., -3.,  8., 6.;
            -2.,  5., -8., 5.;
            -8., -5., -3., 0.;
            -1.,  0., -3., 3.;
        ];

        m.exchange_rows(0, 3);

        //assert!(false, "\n here {} \n", m);

    }



    #[test]
    fn exchange_columns() {

        let mut m: Matrix<f32> = matrix![f32,
            -3., -3.,  8., 6.;
            -2.,  5., -8., 5.;
            -8., -5., -3., 0.;
            -1.,  0., -3., 3.;
        ];

        m.exchange_columns(0, 3);

        //assert!(false, "\n here {} \n", m);
    }


    
    #[test]
    fn id_test() {

        let id: Matrix<f64> = Matrix::id(5);
        
        assert!(id.is_diag(), "\n ID should be diagonal {} \n \n", id);
    }
    


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



    //#[test]
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
    
        let (A11, A12, A21, A22) = decompose_blocks(&t);
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



    //#[test]
    fn strassen_test() {
        
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
        
        let C = strassen(&A, &B);
        
        let s = format!("\n [{}] result \n {:?} \n", C.sum(), C);
        
        println!("{}", s);

        let equal = eq_bound_eps(&expected, &C);

        assert!(equal, "should be equal");
    }
}

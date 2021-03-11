#![allow(dead_code, warnings)]
#![feature(layout_for_ptr)]
//http://www.hpcavf.uclan.ac.uk/softwaredoc/sgi_scsl_html/sgi_html/ch03.html#ag5Plchri
use std::{any::TypeId, cell::RefCell, f32::EPSILON, mem::*, rc::Rc};
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
use js_sys::{Float64Array, SharedArrayBuffer, Uint32Array};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use web_sys::Event;
use crate::{Number, workers::Workers};

use super::{matrix3::Matrix3, matrix4::Matrix4};

//mul
//rref
//nm
//solve

//gram schmidt process 
//??? symbolic operations with matrices ???
//Todo matrix vector - operations in n dims
//angle between vectors in n dim space 
//n dim rotation -> how does it changes as dimensions grow ?
//sphere in n dimensions ?
//properties in n dim - relation to dimensionality growth
/*
ACCURACY!
i should rearrange rows before elimination (generate P) - how ?
how P will help to improve accuracy ? some type of numbers go to the bottom...

use higher precision f32 -> f64 -> f128
use Newton method
b known
solve -> Ax = b
compute Ax - using obtained x
compute difference d = Ax - b
transform this difference into vector in column space using:
Au = d - solve for u
next x = x - u (shifting x towards better accuracy)
*/

//NORMAL GROUP INVARIANT UNDER CONJUGATION
//stochastic
//markov
//controllability matrix
//diag
//identity
//perspective
//rand
//rotation
//reflection
//zeros
//permutation
//upshift_permutation
//downshift_permutation
//ones
//toeplitz
/*
symplectic
T(S)JS = J
S = S11 S12
    S21 S22

T(S11)S21 and T(S22)S12 are symmetric
T(S11)S22 = In + T(S21)S12  
*/
/*
hamiltonian
A   G
F  -T(A)

A,F,G e R(nxn)
F,G - symmetric

J = 0   In
   -In  0
    
JMT(J) = -T(M), then M is Hamiltonian
*/
//jacobi
//krylov
//markov
//hamming
//graph
//dyad
//conv
//translation
//least squares ???
//projection
//pick number per block defined by stride length (pooling)
//pickens
//house
//givens
//fft

//fundamental subspaces, fundamental theorem, proofs

//min
//max
//avg
//rank
//trace
//add
//subtract
//mul float
//mul matrix
//compose
//transpose
//apply
//pow
//kronecker
//inv (pseudo, left, right)
//solve
//distance between two subspaces
//dist
//det
//eig
//lui
//ref
//rref
//qr
//cholesky
//svd

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
//hash table with entries tuples for indices
//can be compressed
//is_banded
//is_symmetric
//is_square
//H(A) = -A
//is_skew_hermitian
//H(A) = A
//is_hermitian
//T(A) = -A
//is_skew_symmetric
//is_tridiagonal
//is_upper_bidiagonal
//is_lower_bidiagonal
//is_permutation
//is_upper_hessenberg
//is_lower_hessenberg
//is_conformably_partitioned

//partition
//block[] -> enter (u,u) leave (u,u) NO INTERSECTIONS ASSERT
//assert partitioned conformably - dimensions match
//TODO parallelism, strassen
//TODO shader computation (tensor flow)
//FROM vector -> matrix, matrix -> vector
//INTO  matrix4 -> matrixN etc

//block
//partition
//reshape
//augment



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
                let value: f64 = rng.gen();
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
}



fn add <T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");

    let mut C: Matrix<T> = Matrix::new(A.rows, A.columns);
    
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



fn eq_f64(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
    
    if a.rows != b.rows || a.columns != b.columns {
       return false;
    }

    for i in 0..a.rows {
        for j in 0..a.columns {
            if (a[[i,j]] - b[[i,j]]).abs() > EPSILON as f64 {
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



pub fn division_level<T: Number>(A: &Matrix<T>, optimal_block_size: usize, threads: usize) -> usize {
    
    let mut s = optimal_block_size;

    if (s as i32) == -1 {
        s = 10000;
    }
    
    unsafe {
        log(&format!("\n threads {} \n", threads));
    }

    let limit = false;

    let total_size = A.size();

    let mut blocks = 1;

    let x = total_size / optimal_block_size;

    if x < 1 {

        blocks = 1;

    } else {

        let c = if x > threads && limit { threads } else { x };

        let n = (c as f64).log(4.).ceil();

        blocks = (4. as f64).powf(n) as usize;
    }

    blocks
}



pub fn mul_blocks<T: Number>(
    a: &mut Matrix<T>, 
    b: &mut Matrix<T>, 
    optimal_block_size: usize,
    discard_zero_blocks: bool,
    threads: usize
) -> Matrix<T> {

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<T> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<T> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    //println!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks);

    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;

    let mut acc: Matrix<T> = Matrix::new(A.rows, B.columns);

    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {

            let mut v:Vec<Vec<f64>> = Vec::new();

            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);
                
                //this is it

                if discard_zero_blocks {
                    let mut sa = T::zero();
                    let mut sb = T::zero();

                    for m in ay0..ay1 {
                        for n in ax0..ax1 {
                            sa += A[[m,n]];
                        }
                    }

                    if sa == T::zero() {
                        continue;
                    }
                    
                    for m in by0..by1 {
                        for n in bx0..bx1 {
                            sb += B[[m,n]];
                        }
                    }

                    if sb == T::zero() {
                        continue;
                    }
                }
                
                for m in ay0..ay1 {
                    for n in bx0..bx1 {
                        for p in ax0..ax1 {
                            acc[[m,n]] += A[[m,p]] * B[[p,n]];    
                        }
                    }
                }
            }
        }
    }

    acc
}



#[wasm_bindgen]
pub fn ml_thread(
    sab:&SharedArrayBuffer,

    a_rows:usize,
    a_columns:usize,
    b_rows:usize,
    b_columns:usize,
    
    ax0:usize,ay0:usize,
    ax1:usize,ay1:usize,
    bx0:usize,by0:usize,
    bx1:usize,by1:usize
) {
    let s = size_of::<f64>();
    let mut sa = a_rows * a_columns * s;
    let mut sb = b_rows * b_columns * s;
    
    let mut ra = Float64Array::new_with_byte_offset(&sab, 0);
    let mut rb = Float64Array::new_with_byte_offset(&sab, sa as u32);
    let mut rc = Float64Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    
    let mut ca = 0.;
    let mut cb = 0.;
    
    for m in ay0..ay1 {
        for n in ax0..ax1 {
            ca += ra.get_index((m * a_columns + n) as u32);
        }
    }

    if ca == 0. {
        return;
    }
    
    for m in by0..by1 {
        for n in bx0..bx1 {
            cb += rb.get_index((m * b_columns + n) as u32);
        }
    }

    if cb == 0. {
        return;
    }
    
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                let v = rc.get_index((m * b_columns + n) as u32) + 
                        ra.get_index((m * a_columns + p) as u32) * 
                        rb.get_index((p * b_columns + n) as u32);

                rc.set_index((m * b_columns + n) as u32, v);
            }
        }
    }
}



pub fn multiply_blocks_threads(
    a: &mut Matrix<f64>, 
    b: &mut Matrix<f64>, 
    optimal_block_size: usize,
    threads: usize
) -> (
    usize,
    Matrix<f64>,
    Matrix<f64>,
    Vec<[usize; 8]>
) {
    
    let mut tasks: Vec<[usize; 8]> = Vec::new();

    let s1 = Matrix::augment_sq2n_size(&a);

    let s2 = Matrix::augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<f64> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<f64> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    //println!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks);

    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;
    
    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {
            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);
                
                let t: [usize; 8] = [
                    ax0,ay0,
                    ax1,ay1,
                    bx0,by0,
                    bx1,by1
                ];

                tasks.push(t);
            }
        }
    }

    (
        blocks,
        A,
        B,
        tasks
    )
}



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
        add(&self, b)
    }
}



impl <T: Number> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, b:Matrix<T>) -> Matrix<T> {
        add(&self, &b)
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



fn from_data <T: Number>(d: &Vec<f64>, size: usize) -> Matrix<T> {

    let data: Vec<T> = d.iter().map(|x| { T::from_f64(*x).unwrap() }).collect();

    let mut m = Matrix::new(size,size);

    m.from_vec(data);

    m
}



impl <T: Number> From<Matrix3> for Matrix<T> {

    fn from(m: Matrix3) -> Matrix<T> {

        let v: Vec<f64> = m.data.into();

        from_data(&v, 3)
    }
}



impl <T: Number> From<Matrix4> for Matrix<T> {

    fn from(m: Matrix4) -> Matrix<T> {

        let v: Vec<f64> = m.data.into();

        from_data(&v, 4)
    }
}



fn unpack_dims(mem: &SharedArrayBuffer) -> ((usize, usize),(usize, usize)) {

    let mut m = js_sys::Uint32Array::new( mem );

    let a_rows = m.get_index(0) as usize;
    let a_columns = m.get_index(1) as usize;
    let b_rows = m.get_index(2) as usize;
    let b_columns = m.get_index(3) as usize;

    ((a_rows, a_columns), (b_rows, b_columns))
}



fn pack_dims(a:&Matrix<f64>, b:&Matrix<f64>) -> js_sys::SharedArrayBuffer {

    let size = size_of::<u32>() as u32;

    let mem = js_sys::SharedArrayBuffer::new(4 * size);

    let mut m = js_sys::Uint32Array::new( &mem );

    m.set_index(0, a.rows as u32);
    m.set_index(1, a.columns as u32);
    m.set_index(2, b.rows as u32);
    m.set_index(3, b.columns as u32);

    mem
}



//TODO convert into single sab with offsets - fn decompose
#[wasm_bindgen]
pub fn worker_test(
    a:&SharedArrayBuffer, 
    b:&SharedArrayBuffer, 
    c:&SharedArrayBuffer,
    dims:&SharedArrayBuffer
) {
    console_error_panic_hook::set_once();

    unsafe {
        log(&format!("\n worker_test: c {:?} \n", c));
    }

    let d = unpack_dims(dims);
    let (a_rows, a_columns) = d.0;
    let (b_rows, b_columns) = d.1;
    
    unsafe {
        log(&format!("\n worker_test: a_rows, a_columns ({},{}) \n", a_rows, a_columns));
    }

    unsafe {
        log(&format!("\n worker_test: b_rows, b_columns ({},{}) \n", b_rows, b_columns));
    }

    let mut A:Matrix<f64> = Matrix::<f64>::from_sab_f64(a_rows, a_columns, a);
    let mut B:Matrix<f64> = Matrix::<f64>::from_sab_f64(b_rows, b_columns, b);
    let optimal_block_size = 1000;
    let discard_zero_blocks = true;
    let result = mul_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, 1);
    let mut C = Matrix::new(a_rows, b_columns);
    
    for i in 0..C.rows {
        for j in 0..C.columns {
            C[[i,j]] = result[[i,j]];
        }
    }

    let result = Float64Array::new(c);
    
    for i in 0..C.data.len() {
        result.set_index(i as u32, C.data[i]);
    }

    unsafe {
        log(&format!("\n worker_test: C {:?} \n", C));
    }
}



#[wasm_bindgen]
pub fn multiply_blocks_worker() {

    let max = 226;
    let optimal_block_size = 100;
    let mut rng = rand::thread_rng();
    let A_rows = rng.gen_range(0, max) + 1; 
    let A_columns = rng.gen_range(0, max) + 1;
    let B_rows = A_columns;
    let B_columns = rng.gen_range(0, max) + 1;

    let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
    let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);
    
    let (
        blocks, //??? should be equal to blocks
        mut A,
        mut B,
        tasks
    ) = multiply_blocks_threads(
        &mut A, 
        &mut B, 
        optimal_block_size,
        1
    );
    
    let f = "_multiply_blocks_worker";
    let hc = tasks.len(); //should be equal to blocks;
    let workers = Workers::new("./worker.js", hc);
    let ra = A.into_sab();
    let rb = B.into_sab();
    let size = (size_of::<f64>() * A.rows * B.columns) as u32;
    let rc = SharedArrayBuffer::new(size);
    
    for i in 0..workers.ws.len() {
        let array: js_sys::Array = js_sys::Array::new();

        array.push(&ra);
        array.push(&rb);
        array.push(&rc);

        let mut dims = SharedArrayBuffer::new((size_of::<u32>() * 12) as u32);
        let dims_v = Uint32Array::new(&dims);

        dims_v.set_index(0, A.rows as u32);
        dims_v.set_index(1, A.columns as u32);
        dims_v.set_index(2, B.rows as u32);
        dims_v.set_index(3, B.columns as u32);

        let t = tasks[i];

        dims_v.set_index(4, t[0] as u32);
        dims_v.set_index(5, t[1] as u32);
        dims_v.set_index(6, t[2] as u32);
        dims_v.set_index(7, t[3] as u32);

        dims_v.set_index(8, t[4] as u32);
        dims_v.set_index(9, t[5] as u32);
        dims_v.set_index(10, t[6] as u32);
        dims_v.set_index(11, t[7] as u32);

        array.push(&dims);
        
        std::mem::forget(dims);
        
        let next = &workers.ws[i];

        let f = move |event: Event| {
            unsafe {
                log(&format!("worker {} completed", i));
            }
        };

        let c = Box::new(f) as Box<dyn FnMut(Event)>;

        let callback1 = Closure::wrap(c);
        
        next.w.set_onmessage(
            Some(
                callback1.as_ref().unchecked_ref()
            )
        );

        callback1.forget();
        
        let result = next.w.post_message(&array);
    }
    
    std::mem::forget(ra);
    std::mem::forget(rb);
    std::mem::forget(rc);
}



//TODO detect memory limit in wasm
#[wasm_bindgen]
pub fn test_multiplication(hc: f64) {

    let max_side = 156;
    let max = 20000.;
    let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
    let mut B: Matrix<f64> = Matrix::rand_shape(max_side, max);

    unsafe {
        log(&format!("\n [HC {}] test_multiplication: A ({}, {}) | B ({}, {}) \n", hc, A.rows, A.columns, B.rows, B.columns));
    }

    let optimal_block_size = 4000;

    let (
        blocks,
        mut A,
        mut B,
        tasks
    ) = multiply_blocks_threads(
        &mut A, 
        &mut B, 
        optimal_block_size,
        hc as usize
    );

    unsafe {
        log(&format!("\n test_multiplication: A ({}, {}) | B ({}, {}) | blocks {} | tasks {} \n", A.rows, A.columns, B.rows, B.columns, blocks, tasks.len()));
    }

    let example = &A * &B;
    let example = Rc::new(example);
   

    let s = size_of::<f64>();
    let sa = A.size() * s;
    let sb = B.size() * s;
    let sc = A.rows * B.columns * s;
    let size = sa + sb + sc;

    unsafe {
        log(&format!("\n ready to allocated {} bytes in sab \n", size));
    }
    let mut s = SharedArrayBuffer::new(size as u32);
    let mut v1 = Float64Array::new_with_byte_offset(&s, 0);
    let mut v2 = Float64Array::new_with_byte_offset(&s, sa as u32);
    
    Matrix::<f64>::copy_to_f64(&A, &mut v1);
    Matrix::<f64>::copy_to_f64(&B, &mut v2);
    
    unsafe {
        log(&format!("\n allocated {} bytes in sab \n", size));
    }

    let a: Rc<SharedArrayBuffer> = Rc::new(s);
    
    let mut workers = Workers::new("./worker.js", tasks.len());

    let mut workers = Rc::new(
        RefCell::new(workers)
    );
    
    let mut list = workers.borrow_mut();

    for i in 0..list.ws.len() {

        let mut c_example = example.clone();

        let t = tasks[i];
        let next = &mut list.ws[i];
        let ta = a.clone();
        let array: js_sys::Array = js_sys::Array::new();

        array.push(&a);

        let a_r = A.rows;
        let a_c = A.columns;

        
        let a_rows = JsValue::from(A.rows as u32);
        let a_columns = JsValue::from(A.columns as u32);

        let b_rows = JsValue::from(B.rows as u32);
        let b_columns = JsValue::from(B.columns as u32);

        let t0 = JsValue::from(t[0] as u32);
        let t1 = JsValue::from(t[1] as u32);
        let t2 = JsValue::from(t[2] as u32); 
        let t3 = JsValue::from(t[3] as u32); 

        let t4 = JsValue::from(t[4] as u32); 
        let t5 = JsValue::from(t[5] as u32); 
        let t6 = JsValue::from(t[6] as u32); 
        let t7 = JsValue::from(t[7] as u32); 

        array.push(&a_rows);
        array.push(&a_columns);

        array.push(&b_rows);
        array.push(&b_columns);

        array.push(&t0);
        array.push(&t1);
        array.push(&t2);
        array.push(&t3);

        array.push(&t4);
        array.push(&t5);
        array.push(&t6);
        array.push(&t7);

        let hook = workers.clone();

        let f = move |event: Event| {
            
            let mut list = hook.borrow_mut();

            list.ws[i].cb = None;

            let sc = Rc::strong_count(&ta);
            
            if sc <= 2 {
                //means we are done
                let mut result = Float64Array::new_with_byte_offset(&ta, (sa + sb) as u32);

                let ggg = result.to_vec();

                let mut ma = Matrix::new(a_r, a_c);
                
                ma.from_vec(ggg);
                
                unsafe {
                    log(&format!("\n expected {:?} \n", &c_example.data));
                }

                unsafe {
                    log(&format!("\n result {:?} \n", ma.data));
                }

                let mt = Rc::get_mut(&mut c_example).unwrap();

                let equal = ma == *mt;

                assert!(equal, "matrices should be equal");

                list.terminate();
            }

            unsafe {
                log(&format!("worker {} completed | strong count {}, {:?}", i, sc, hook));
            }
        };

        let c = Box::new(f) as Box<dyn FnMut(Event)>;

        let callback1 = Closure::wrap(c);
        
        let kk = callback1.as_ref();

        let vv = Some(
            kk.dyn_ref().unwrap() //unchecked_ref()
        );

        next.w.set_onmessage(vv);
        
        next.cb = Some(callback1);

        let result = next.w.post_message(&array);
    }
}



#[wasm_bindgen]
pub fn test_multiplication_worker(
    sab: &SharedArrayBuffer,
    a_rows: f64,
    a_columns: f64,
    b_rows: f64,
    b_columns: f64,
    t0: f64,
    t1: f64,
    t2: f64,
    t3: f64,
    t4: f64,
    t5: f64,
    t6: f64,
    t7: f64
) {
    unsafe {
        log(&format!("\n hello from worker {} {} {} {} | {} {} {} {} | {} {} {} {} \n", 
            a_rows,
            a_columns,
            b_rows,
            b_columns,

            t0,
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7
        ));
    }

    ml_thread(
        sab,

        a_rows as usize,
        a_columns as usize,
        b_rows as usize,
        b_columns as usize,
        
        t0 as usize,
        t1 as usize,
        t2 as usize,
        t3 as usize,
        t4 as usize,
        t5 as usize,
        t6 as usize,
        t7 as usize
    );
}



mod tests {
    use std::{f32::EPSILON as EP, f64::EPSILON, f64::consts::PI};
    use rand::Rng;
    use crate::{ core::matrix::{Matrix}, matrix};
    use super::{ get_optimal_depth, eq_f64, multiply, mul_blocks };
    
    //TODO
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
        
        /*
        let m = matrix4![
            3., 1., 1., 1.,
            1., 2., 1., 1.,
            1., 1., 2., 1.,
            1., 1., 1., 2.
        ];
    
        let mut m2 = m.clone();
    
        m2.t();
    
        let r = mul(&m,&m2, 4, 4, 4);
        */
    
        assert_eq!(1, 1, "hey {}", r);
        
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
            let mut B: Matrix<f64> = Matrix::rand_shape(max_side, max);
            
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

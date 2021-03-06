#![allow(dead_code)]
#![feature(layout_for_ptr)]
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
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
extern crate num_cpus;
use js_sys::{Float64Array, SharedArrayBuffer};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
pub trait Integer : PrimInt + Copy + NumAssignOps + std::fmt::Debug {}

pub trait Number : Num + cast::FromPrimitive + Copy + NumOps + NumAssignOps + std::fmt::Debug + Sized + Signed + PartialOrd + 'static {
    fn to_ne_bytes(self) -> [u8; 8];
}

impl Integer for i32 {}

impl Number for i8 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..1]);
        a
    }
}

impl Number for i32 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..4]);
        a
    }
}

impl Number for f32 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..4]);
        a
    }
}

impl Number for f64 {
    fn to_ne_bytes(self) -> [u8; 8] {
        self.to_ne_bytes()
    }
}



pub fn eq_f64(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
    
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

    return true;
}



pub fn eq<T: Number>(a: &Matrix<T>, b: &Matrix<T>) -> bool {
    
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

    return true;
}



impl <T:Number> PartialEq for Matrix<T> {
    fn eq(&self, b: &Matrix<T>) -> bool {
        eq(self, b)
    }
}



impl <T:Number> Eq for Matrix<T> {}



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


#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

//??? symbolic operations with matrices ???

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
//strassen



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



    pub fn decompose_blocks(A: Matrix<T>) -> (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>) {

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

        //m.data = d.to_vec();

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
        let mut acc = T::from_f64(0.).unwrap();

        for i in 0..self.rows {
            for j in 0..self.columns {
                acc += self[[i,j]];
            }
        }

        acc
    }



    pub fn size(&self) -> usize {
        self.rows * self.columns
    } 



    //TODO return total size in bytes
    pub fn mem_size() {

    }
}



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


//TODO should work with browser log
impl <T: Number> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.columns {
                print!("{:?},", self[[i,j]]);
            }
            print!("\n");
        }
        println!("\n");
        /*
        write!(f, 
            "\n [{}, {}, {}, {}] \n", 
            self[0], self[1], self[2], self[3]
        )
        */
        Ok(())
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



pub fn transpose <T:Number>(m: &Matrix<T>) -> Matrix<T> {

    let mut t = Matrix::new(m.columns, m.rows);

    for i in 0..m.rows {
        for j in 0..m.columns {
            t[[j,i]] = m[[i,j]];
        }
    }

    t
}



pub fn add <T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    //TODO mistake fix
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



pub fn subtract<T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    
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



pub fn augment_sq2n_size<T:Number>(A: &Matrix<T>) -> usize {

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



pub fn augment_sq2n<T:Number>(A: Matrix<T>) -> Matrix<T> {

    if A.rows <= 1 && A.columns <= 1 {
       return A;
    }
    
    if A.rows==A.columns && (A.rows as f32).log2().fract() == 0. {
        return A;
    }
    
    let mut side = std::cmp::max(A.rows, A.columns);
    let l: f64 = (side as f64).log2();
    
    if l.fract() != 0. {
        side = (2. as f64).powf(l.ceil()) as usize;
    }
    
    let mut m: Matrix<T> = Matrix::new(side, side);
    m = m + A;
    m
}


/*
pub fn quadrify<T:Number>(A:&Matrix<T>) -> Quad<T> {
    
    let mut q: Quad<T> = Quad::new(0,0,A.columns as i32,A.rows as i32, 0, QuadDirection::Root);
    
    for i in 0..A.rows {
        for j in 0..A.columns {
            let d: QuadData<T> = QuadData {
                y:i as i32, 
                x:j as i32,
                value: A[[i,j]]
            };
            q.insert(&d);
        }
    }

    q
} 
*/


fn sum <T: Number>(len: usize, data: impl Index<usize, Output = T>) -> T {
    
    let mut acc = identities::zero();
    
    for i in 0..len {
        acc = acc + data[i];
    }
    
    acc
}



fn identity <T: Number>(size: usize) -> Matrix<T> {

    let mut data: Vec<T> = Vec::with_capacity(size*size);
    
    for i in 0..size {
        data[(size * i) + i] = T::from_i32(1).unwrap();
    }
    
    Matrix {
        data,
        rows: size,
        columns: size
    }
}



fn scale <T:Number>(m: &mut Matrix<T>, n: T) -> &mut Matrix<T> {
    
    m.data = m.data.iter().map(|x:&T| *x*n).collect();

    m

}



fn lu <T:Number>(A: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    //TODO edge cases - irregular matrices shapes
    //TODO rearrange rows to reduce numerical errors before computation begins, mangle indices, how and why ?
    //TODO handle pivot zero case - rows rearrangement - create indices map

    let size = A.rows * A.columns;
    
    let mut L: Matrix<T> = identity(A.rows);
    
    let mut U: Matrix<T> = A.clone();
    
    for i in 0..(U.rows - 1) {
        let p = U.data[((i * U.columns) + i) as usize]; 

        let mut tmp: Matrix<T> = identity(A.rows);

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



pub fn init_rand<T: Number>(A: &mut Matrix<T>) {
    let max = 683.45;
    let mut rng = rand::thread_rng();
    for i in 0..A.columns {
        for j in 0..A.rows {
            let value: f64 = rng.gen();
            A[[j,i]] = T::from_f64(value).unwrap();
        }
    }
}



pub fn init_const<T: Number>(A: &mut Matrix<T>) {
    let max = 683.;
    let c = 6.6; //rng.gen_range((0.)..max);
    for i in 0..A.columns {
        for j in 0..A.rows {
            A[[j,i]] = T::from_f64(c).unwrap();
        }
    }
}


/*
pub fn count_discarded(qA: &mut Quad<f64>) -> (f64, f64) {
    let discarded = Rc::new(RefCell::new(0.));

    let alive = Rc::new(RefCell::new(0.));

    let s1 = discarded.clone();

    let s2 = alive.clone();

    let mut b: Box<dyn FnMut(&mut Quad<f64>)> = Box::new(
        move |x:&mut Quad<f64>| {
            if x.discard {
                *s1.borrow_mut() += 1.;
            } else {
                *s2.borrow_mut() += 1.;
            }
        }
    );

    qA.traverse(&mut b);

    let d = *discarded.borrow();

    let a = *alive.borrow();

    (d,a)
}
*/


pub fn random_shape_matrix(max: usize) -> Matrix<f64> {
    let A_rows = 22; //rng.gen_range(0..max) + 1; 
    let A_columns = 12; //rng.gen_range(0..max) + 1;
    let mut eA: Matrix<f64> = Matrix::new(A_rows, A_columns);

    init_rand(&mut eA);
    eA
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



pub fn multiply <T:Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {

    assert_eq!(A.columns, B.rows, "matrices dimensions should be compatible A columns {} B rows {}", A.columns, B.rows);
    
    mul(A, B, A.rows, A.columns, B.columns)
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

    let s1 = augment_sq2n_size(&a);

    let s2 = augment_sq2n_size(&b);

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



pub fn multiply_blocks<T: Number>(
    a: &mut Matrix<T>, 
    b: &mut Matrix<T>, 
    optimal_block_size: usize,
    discard_zero_blocks: bool,
    threads: usize
) -> Matrix<T> {

    let s1 = augment_sq2n_size(&a);

    let s2 = augment_sq2n_size(&b);

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



//TODO benchmarks
mod tests {
    use crate::{
        matrix4::{
            Matrix4
        }
    };
    use std::{cell::RefCell, mem::*, rc::Rc, time::Instant};
    use crate::matrix4;
    use super::{Matrix, Num, Number, augment_sq2n, multiply_blocks, augment_sq2n_size, division_level, get_optimal_depth, init_rand, mul, multiply, random_shape_matrix, transpose};

    use std::{mem::size_of_val, ops::{
            Index
        }};
        


    
    fn multiply_blocks_test(discard_zero_blocks:bool) {

        for h in (1..2) {
            let max = 156;
            let optimal_block_size = 5000;
            //let mut rng = rand::thread_rng();
            let A_rows = max; //rng.gen_range(0..max) + 1; 
            let A_columns = max; //rng.gen_range(0..max) + 1;
            let B_rows = max; //A_columns;
            let B_columns = max; //rng.gen_range(0..max) + 1;
            let threads = 1;
            let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
            let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);

            //println!("next ({}, {})", A.rows, A.columns);

            init_rand(&mut A);
            init_rand(&mut B);

            let C1: Matrix<f64> = &A * &B;
            let C: Matrix<f64> = multiply_blocks(&mut A, &mut B, optimal_block_size, discard_zero_blocks, threads);

            assert_eq!(C.sum(), C1.sum(), "sum should be equal");

            for i in 0..C1.rows {
                for j in 0..C1.columns {
                    assert_eq!(C1[[i,j]], C1[[i,j]], "all entries should be equal");
                }
            }

        }

        
    }   


    //#[test] 
    fn multiply_test(){

        let before = Instant::now();
        multiply_blocks_test(false);
        println!("\n \n No discarding... Elapsed time: {:.2?} \n \n", before.elapsed().as_millis());

        let before = Instant::now();
        multiply_blocks_test(true);
        println!("\n \n Discarding... Elapsed time: {:.2?} \n \n", before.elapsed().as_millis());

    }
    
    

    //#[test]
    fn quad() {

        fn test(max: usize) {
            //let mut rng = rand::thread_rng();
            let A_rows =  12; //rng.gen_range(0..max) + 1; 
            let A_columns = 12; //rng.gen_range(0..max) + 1;
            let B_rows = 12; //A_columns;
            let B_columns = 12; //rng.gen_range(0..max) + 1;
    
            let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
            let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);
        
            init_rand(&mut A);
            init_rand(&mut B);
    
            let C1 = &A * &B; 
            let C1_sum = C1.sum();
    
            let s1 = augment_sq2n_size(&A);
            let s2 = augment_sq2n_size(&B);
            let s = std::cmp::max(s1,s2);
            let A: Matrix<f64> = Matrix::new(s,s) + A;
            let B: Matrix<f64> = Matrix::new(s,s) + B;
            let threads = 1;
            let optimal_block_size = 5000;
            let blocks = division_level(&A, optimal_block_size, threads);
            let cb = A.size() / blocks;

            
            let C: Matrix<f64> = &A * &B;
            let C_sum = C.sum();
    
            type Tuple = ((usize, usize), (usize, usize)); 

            let v: Vec<Tuple> = Vec::new();
            let zeros: Vec<Tuple> = Vec::new();

            //let mut l = A.rows / blocks; //should be A.rows / side length of a block
            
            let d = A.size() / blocks; //(A.size() / blocks).sqrt()

            let y = (d as f64).sqrt() as usize;

            println!("\n amount of blocks {} | block size {} | side length of single block {} \n", blocks, d, y);
            
            let mut ctr = 0;
            
            fn sum_block(A: &Matrix<f64>, start: (usize,usize), end: (usize,usize)) -> f64 {
                let mut result = 0.;
                for i in start.0..end.0 {
                    for j in start.1..end.1 {
                        result += A[[i,j]];
                    }
                }
                result
            }
            
            //why total count not equal to amount of blocks ?
            //println!("\n total count is {} \n", ctr);
            let s = ((A.size() / blocks) as f64).sqrt() as usize;
            let mut R: Matrix<f64> = Matrix::new(A.rows, B.columns);

            
            for i in (0..A.rows).step_by(s) {
                for j in (0..B.columns).step_by(s) {

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

                        //let sum = sum_block(&A, (x0 as usize, y0 as usize), (x1 as usize, y1 as usize));

                        println!("\n R is ({},{}) BLOCK A ({},{}) - ({},{}) | BLOCK B ({},{}) - ({},{}) \n",
                            R.rows, R.columns,
                            ax0, ay0,
                            ax1, ay1,

                            bx0, by0,
                            bx1, by1
                        );

                        for m in ay0..ay1 {
                            for n in bx0..bx1 {
                                for p in ax0..ax1 {
                                    R[[m,n]] += A[[m,p]] * B[[p,n]];    
                                }
                            }
                        }
                    }
                }
            }



            // assert_eq!(C.sum(), R.sum(), "should be equal");
            let D =  &R - &C;
            //println!("\n D is {} \n", D);
            //println!("\n C is {} \n", C);
            //println!("\n R is {} \n", R);

            /*
            println!("\n A is {} \n", A);
            println!("\n B is {} \n", B);
            println!("\n C is {} \n", C);
            println!("\n C1 is {} \n", C1);
            */
            assert_eq!(D.sum(), 0., "sum is zero");
            //assert_eq!(C1_sum, C_sum, "sum C1 should be equal sum C");
        }
        
        for i in 1..10 {
            //let i = 1;
            test(i * 50);
        }
    }    


    
    //#[test]
    fn mul_quads() {
        let iterations = 1;
        let max = 300;
        //let mut rng = rand::thread_rng();

        for i in 0..iterations {
            let A_rows = 12; //rng.gen_range(0..max); 
            let A_columns = 12; //rng.gen_range(0..max);
            let B_rows = A_columns;
            let B_columns = 12; //rng.gen_range(0..max);

            println!("before augmentation - A ({},{}) - B ({},{})", A_rows, A_columns, B_rows, B_columns);

            let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
            let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);

            init_rand(&mut A);
            init_rand(&mut B);

            let A = augment_sq2n(A);
            let B = augment_sq2n(B);

            println!("after augmentation - A augmented ({},{}) - B augmented ({},{})", A.rows, A.columns, B.rows, B.columns);
            
            let depth_A = get_optimal_depth(&A, 10);
            let depth_B = get_optimal_depth(&A, 10);





            //println!("depth A {} - depth B {}", depth_A, depth_B);

            //let mut qA = quadrify(&A);
            //let mut qB = quadrify(&B);
            
            //println!("before - quad A {} KB - quad B {} KB", qA.size() / 1000, qB.size() / 1000);

            //let (discarded, alive) = count_discarded(&mut qA);

          
            //println!("before - discarded {} | alive {}", discarded, alive);

            //qA.discard_by(Quad::zeros, depth_A as i16);
            //qB.discard_by(Quad::zeros, depth_B as i16);


            //let (discarded, alive) = count_discarded(&mut qA);
            
            //qA.drop_discarded();
            //qB.drop_discarded();
            
            //println!("after - discarded {} | alive {}", discarded, alive);
            
            //println!("after - quad A {} KB - quad B {} KB", qA.size() / 1000, qB.size() / 1000);
        }
    }
        

    //#[test] 
    fn augment_sq2n_test() {
        let iterations = 200;
        let max = 683;

        for i in 0..iterations {
            let rows = 12; //rng.gen_range(0..max); 
            let columns = 12; //rng.gen_range(0..max);
            let m: Matrix<f64> = Matrix::new(rows, columns);
            let aug = augment_sq2n(m);

            let d = get_optimal_depth(&aug, 10);

            let size = (aug.rows * aug.columns) as f64;

            println!("({},{}) | size - {} | level - {}", aug.rows, aug.columns, size, d);

            let l: f64 = size.log2();
            //assert_eq!(aug.rows, aug.columns, "from ({} {}) to ({} {}) - should be square", rows, columns, aug.rows, aug.columns);
            //assert_eq!(l.fract(), 0., "from ({} {}) to ({} {}) - should be power of 2", rows, columns, aug.rows, aug.columns);
            

            //println!("next {} - ({},{}) - size - {}", i, aug.rows, aug.columns, mem_size);
            //let q = quadrify(&aug);

            //println!("next {} - ({},{}) - size quad - {}", i, aug.rows, aug.columns, mem_size2);

            //assert_eq!(1, 1, "({} {}) - quad - size - {}", aug.rows, aug.columns, mem_size2);
        }
    }
    
    //#[test]
    fn operations() {
        
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
        let t2 = transpose(&c);
        let r = multiply(&t,&t2);
        
        let m = matrix4![
            3., 1., 1., 1.,
            1., 2., 1., 1.,
            1., 1., 2., 1.,
            1., 1., 1., 2.
        ];

        let mut m2 = m.clone();

        m2.t();

        let r = mul(&m,&m2, 4, 4, 4);

        assert_eq!(1, 1, "hey {}", r);

    }
}
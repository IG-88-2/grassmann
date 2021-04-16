#![feature(layout_for_ptr)]
use std::{any::TypeId, cell::RefCell, cmp::min, f32::EPSILON, f64::NAN, future::Future, mem::*, pin::Pin, rc::Rc, task::{Context, Poll}, time::Instant};
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
use crate::{Number, Partition, vector, workers::Workers};
use crate::{functions::{ 
    rand::{ rand },
    rank::{ rank },
    rand_shape::{ rand_shape },
    rand_diag::{ rand_diag },
    rand_sing2::{ rand_sing2 },
    rand_sing::{ rand_sing },
    depth::{ depth },
    augment_sq2n_size::{ augment_sq2n_size },
    augment_sq2n::{ augment_sq2n },
    set_diag::{ set_diag },
    sum::{ sum },
    trace::{ trace },
    round::{ round }, 
    givens_theta::{ givens_theta }, 
    project::{ project }, 
    upper_hessenberg::{ upper_hessenberg }, 
    lower_hessenberg::{ lower_hessenberg }, 
    is_upper_hessenberg::{ is_upper_hessenberg }, 
    is_lower_hessenberg::{ is_lower_hessenberg }, 
    init_const::{ init_const }, 
    transpose::{ transpose }, 
    id::{ id }, 
    copy_to_f64::{ copy_to_f64 }, 
    from_sab_f64::{ from_sab_f64 }, 
    transfer_into_sab::{ transfer_into_sab }, 
    into_sab::{ into_sab }, 
    inv_diag::{ inv_diag }, 
    inv_lower_triangular::{ inv_lower_triangular }, 
    inv_upper_triangular::{ inv_upper_triangular }, 
    inv::{ inv }, 
    perm::{ perm }, 
    rand_perm::{ rand_perm },
    partition::{ partition }, 
    assemble::{ assemble }, 
    cholesky::{ cholesky }, 
    schur_complement::{ schur_complement }, 
    lps::{ lps }, 
    is_diagonally_dominant::{ is_diagonally_dominant }, 
    is_identity::{ is_identity }, 
    is_diag::{ is_diag }, 
    is_upper_triangular::{ is_upper_triangular }, 
    is_lower_triangular::{ is_lower_triangular }, 
    is_permutation::{ is_permutation }, 
    is_symmetric::{ is_symmetric }, 
    into_basis::{ into_basis }, 
    from_basis::{ from_basis }, 
    eig::{ eig, eigenvectors, eig_decompose }, 
    lu::{ block_lu, block_lu_threads_v2, lu, lu_v2 },
    conjugate::{ conjugate },
    svd::{ svd_jac1 }, 
    qr::{
        qr, givens_qr, apply_Q_givens_hess, apply_Qt_givens_hess, givens_qr_upper_hessenberg, form_Qt_givens_hess, 
        form_Q_givens_hess, apply_q_R, form_Q, form_P, house_qr, apply_q_b
    }, 
    solve::{ solve_upper_triangular, solve, solve_lower_triangular }, 
    utils::{eq_bound_eps, eq_bound, eq_bound_eps_v, eq_eps_f64}
}};

use super::{matrix3::Matrix3, matrix4::Matrix4, vector::Vector};



//gemm
//debugger
//inverse (pseudo, left, right)
//matrix norm
//matrix norm 2 norm
//min
//max
//avg
//off diag sum
//numerical integration
//numerical differentiation
//jacobian
//convolutions
//pooling (pick number per block defined by stride length)
//pickens
//n dim rotation (Jacobi composition)
//foreign function interface - call C routines
//vandermonde
//projection - properties and geometry of higher dimensional spaces (distance between two subspaces, Cauchy–Schwarz inequality in n-dim)
//action on a sphere
//sparse (hash table with entries tuples for indices)
//complex numbers
//monte carlo
//kalman filter
//fft
//wavelets
//wedge
//cramer (co-factors)
//markov
//krylov subspace
//controllability matrix
//hamming
//graph
//kronecker product
//update lu
//extract submatrix - takes (i,j) matrix first or last entry and dimensions
//drop row
//drop column
//drop sub diagonal
//drop cross



/*
use Newton method
b known
compute Ax - using obtained x
compute difference d = Ax - b
transform this difference into vector in column space using:
Au = d - solve for u
next x = x - u (shifting x towards better accuracy)
*/



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



#[macro_export]
macro_rules! compose_m {
    ($v:expr) => { $v };
    ($v:expr, $($x:expr),+) => {

        &( $v ) * &( compose_m!($($x),*) )

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
    


    //TODO lu should keep track and return info about sign of det (n of perm)
    pub fn det(&self) -> f64 {

        let lu = self.lu();

        println!("\n det L {}, U {} \n", lu.L, lu.U);

        let mut acc = 1.;

        for i in 0..lu.U.rows {
            let c = lu.U[[i,i]];
            acc = acc * T::to_f64(&c).unwrap();
        } 

        acc
    }



    pub fn eig2x2(m:&Matrix<f64>) -> Option<(f64, f64)> {

        let t = m.trace();
        
        let a = m[[0, 0]]; 
        let d = m[[1, 1]];
        let b = m[[0, 1]];
        let c = m[[1, 0]];

        let d = a * d - b * c;

        let r = t.powf(2.) - 4. * d;

        if r < 0. {
            return None;
        }

        let y1 = (t + r.sqrt()) / 2.;
        let y2 = (t - r.sqrt()) / 2.;

        Some((y1, y2))
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
    


    pub fn extract_column(&self, i: usize) -> Vector<T> {

        let zero = T::from_f64(0.).unwrap();

        let mut c = vec![zero; self.rows]; 

        for j in 0..self.rows {
            c[j] = self[[j, i]];
        }

        Vector::new(c)
    }
    


    pub fn extract_diag(&self) -> Vector<T> {

        let zero = T::from_f64(0.).unwrap();

        let mut v = vec![zero; self.rows];

        for i in 0..self.rows {
            v[i] = self[[i, i]];
        } 

        Vector::new(v)
    }


    
    pub fn perm(size: usize) -> Vec<Matrix<f32>> {
        
        perm(size)

    }


    
    pub fn inv(&self, lu: &lu<T>) -> Option<Matrix<T>> {
        
        inv(self, lu)

    }



    pub fn lps(&self, n: usize) -> Matrix<T> {
        
        lps(self, n)

    }



    pub fn assemble(p: &Partition<T>) -> Matrix<T> {

        assemble(p)

    }



    pub fn partition(&self, r: usize) -> Option<Partition<T>> {
        
        partition(self, r)

    }



    pub fn rank(&self) -> u32 {

        rank::<T>(self)

    }



    pub fn rand_perm(size: usize) -> Matrix<T> {

        rand_perm(size)

    }



    pub fn apply(&mut self, f: &dyn Fn(T) -> T) {

        self.data = self.data.iter().map(|x:&T| f(*x)).collect();

    }
    
    

    pub fn inv_upper_triangular(&self) -> Option<Matrix<T>> {

        inv_upper_triangular(self)

    }



    pub fn inv_lower_triangular(&self) -> Option<Matrix<T>> {

        inv_lower_triangular(self)

    }



    pub fn inv_diag(&self) -> Matrix<T> {

        inv_diag(self)

    }



    pub fn into_sab(&mut self) -> SharedArrayBuffer {

        into_sab::<T>(self)

    }



    pub fn transfer_into_sab(A:&Matrix<f64>, B:&Matrix<f64>) -> SharedArrayBuffer {
        
        transfer_into_sab(A, B)

    }



    pub fn from_sab_f64(rows: usize, columns: usize, data: &SharedArrayBuffer) -> Matrix<f64> {

        from_sab_f64(rows, columns, data)

    }
    


    pub fn copy_to_f64(m: &Matrix<f64>, dst: &mut Float64Array) {

        copy_to_f64(m, dst)

    }



    pub fn id(size: usize) -> Matrix<T> {

        id(size)

    }



    pub fn transpose(&self) -> Matrix<T> {

        transpose(self)

    }



    pub fn init_const(&mut self, c: T) {
        
        init_const(self, c)

    }



    pub fn rand(rows: usize, columns: usize, max: f64) -> Matrix<T> {

        rand(rows, columns, max)

    }



    pub fn rand_shape(max_side: usize, max:f64) -> Matrix<T> {
        
        rand_shape(max_side, max)

    }



    pub fn rand_diag(size: usize, max: f64) -> Matrix<T> {

        rand_diag(size, max)

    }



    pub fn rand_sing2(size: usize, n: usize, max: f64) -> Matrix<T> {
        
        rand_sing2(size, n, max)

    }



    pub fn rand_sing(size: usize, max:f64) -> Matrix<T> {
        
        rand_sing(size, max)

    }



    pub fn depth(&self) -> i32 {
        
        depth::<T>(self)

    }



    pub fn augment_sq2n_size(&self) -> usize {

        augment_sq2n_size::<T>(self)

    }



    pub fn augment_sq2n(&self) -> Matrix<T> {

        augment_sq2n(self)

    }



    pub fn set_vec(&mut self, v:Vec<T>) {

        assert_eq!(v.len(), self.rows * self.columns, "from_vec - incorrect size");

        self.data = v;
    }



    pub fn set_diag(&mut self, v:Vector<T>) {

        set_diag::<T>(self, v)

    }
    


    pub fn sum(&self) -> T {

        sum(self)

    }



    pub fn trace(&self) -> T {

        trace(self)

    }
    


    pub fn round(&mut self, precision: f64) {

        round::<T>(self, precision)

    }



    pub fn givens_theta(&self, i: usize, j: usize) -> f64 {

        givens_theta::<T>(self, i, j)

    }
    


    pub fn project(&self, b:&Vector<T>) -> Vector<T> {

        project(self, b)

    }



    pub fn size(&self) -> usize {

        self.rows * self.columns

    } 


    
    pub fn mem_size(&self) -> usize {

        size_of::<T>() * self.rows * self.columns

    }



    pub fn upper_hessenberg(&self) -> Matrix<T> {
        
        upper_hessenberg(self)

    }



    pub fn lower_hessenberg(&self) -> Matrix<T> {

        lower_hessenberg(self)

    }



    pub fn is_upper_hessenberg(&self) -> bool {
        
        is_upper_hessenberg::<T>(self)

    }



    pub fn is_lower_hessenberg(&self) -> bool {

        is_lower_hessenberg::<T>(self)

    }



    pub fn is_diagonally_dominant(&self) -> bool {

        is_diagonally_dominant::<T>(self)

    }



    pub fn is_identity(&self) -> bool {

        is_identity::<T>(self)

    }



    pub fn is_diag(&self) -> bool {

        is_diag::<T>(self)

    }



    pub fn is_upper_triangular(&self) -> bool {
        
        is_upper_triangular::<T>(self)

    }



    pub fn is_lower_triangular(&self) -> bool {

        is_lower_triangular::<T>(self)

    }



    pub fn is_square(&self) -> bool {

        self.columns == self.rows

    }



    pub fn is_permutation(&self) -> bool {
        
        is_permutation::<T>(self)

    }



    pub fn is_symmetric(&self) -> bool {

        is_symmetric::<T>(self)

    }



    pub fn is_zero(&self) -> bool {

        let zero = T::from_f64(0.).unwrap();

        self.data.iter().all(|x| { *x == zero })

    } 



    pub fn from_basis(b: Vec<Vector<T>>) -> Matrix<T> {
        
        from_basis(b)

    }



    pub fn into_basis(&self) -> Vec<Vector<T>> {

        into_basis(self)

    }



    pub fn conjugate(&self, S: &Matrix<T>) -> Option<Matrix<T>> {

        conjugate(self, S)
        
    }



    pub fn schur_complement(p:&Partition<T>) -> Option<Matrix<T>> {

        schur_complement(p)

    }

    

    pub fn cholesky(&self) -> Option<Matrix<T>> {

        cholesky(self)

    }



    pub fn solve(&self, b: &Vector<T>, lu: &lu<T>) -> Option<Vector<T>> {

        solve(b, lu, f32::EPSILON as f64)

    }



    pub fn lu(&self) -> lu<T> {

        let mut lu = lu_v2::<T>(self, true, true);

        lu.unwrap()

    }



    pub fn block_lu(&self) -> lu<T> {

        let mut lu = block_lu::<T>(self);

        lu.unwrap()

    }
    


    pub fn qr(&self) -> qr<T> {
        //givens_qr
        house_qr::<T>(self)

    }

    

    pub fn eigenvectors(&self, q: &Vec<f64>) -> Vec<(f64, Vector<T>)> {

        eigenvectors(self, q) 
        
    }


    
    pub fn eig(&self, precision: f64, steps: i32) -> Vec<f64> {

        eig::<T>(self, precision, steps)

    }



    pub fn eig_decompose(&self) -> Option<(Matrix<T>, Matrix<T>, Matrix<T>)> {

        eig_decompose(self)

    }



    pub fn svd(mut self, eps: f64) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        
        svd_jac1(self, eps)

    }
}



pub fn add <T: Number>(A: &Matrix<T>, B: &Matrix<T>, offset:usize) -> Matrix<T> {
    
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");
    
    let mut C: Matrix<T> = A.clone(); //Matrix::new(A.rows, A.columns);
    
    /*
    if offset > 0 {
        for i in 0..offset {
            for j in 0..offset {
                C[[i, j]] = A[[i,j]];
            }
        }
    }
    */

    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i + offset, j + offset]] += B[[i,j]];
            //C[[i + offset, j + offset]] = A[[i,j]] + B[[i,j]];
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
        add(&self, b, 0)
    }
}



impl <T: Number> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, b:Matrix<T>) -> Matrix<T> {
        add(&self, &b, 0)
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



impl <T: Number> Mul <T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(mut self, s:T) -> Matrix<T> {
        let mut A = self.clone();
        scale(&mut A, s);
        A
    }
}



impl <T: Number> Mul <T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(mut self, s:T) -> Matrix<T> {
        scale(&mut self, s);
        self
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



fn from_data <T: Number> (s: usize, d: Vec<f64>) -> Matrix<T> {

    let data: Vec<T> = d.iter().map(|x| { T::from_f64(*x).unwrap() }).collect();

    let mut m = Matrix::new(s, s);

    m.set_vec(data);

    m
}



fn from_vector <T: Number> (v: &Vector<T>) -> Matrix<T> {

    let mut m = Matrix::new(v.data.len(), 1);

    for i in 0..v.data.len() {
        m[[i, 0]] = v[i];
    }
    
    m
}



impl <T:Number> From<Vector<T>> for Matrix<T> {
    fn from(v: Vector<T>) -> Matrix<T> {
        
        from_vector(&v)
    }
}



impl <T:Number> From<&Vector<T>> for Matrix<T> {
    fn from(v: &Vector<T>) -> Matrix<T> {
        
        from_vector(v)
    }
}



impl <T: Number> From<Matrix3> for Matrix<T> {

    fn from(m: Matrix3) -> Matrix<T> {

        let d: Vec<f64> = m.data.into();

        from_data(3, d)
    }
}



impl <T: Number> From<Matrix4> for Matrix<T> {

    fn from(m: Matrix4) -> Matrix<T> {

        let d: Vec<f64> = m.data.into();

        from_data(4, d)
    }
}



impl <T: Number> From<&Matrix3> for Matrix<T> {

    fn from(m: &Matrix3) -> Matrix<T> {

        let d: Vec<f64> = m.data.into();

        from_data(3, d)
    }
}



impl <T: Number> From<&Matrix4> for Matrix<T> {

    fn from(m: &Matrix4) -> Matrix<T> {

        let d: Vec<f64> = m.data.into();

        from_data(4, d)
    }
}



impl <T:Number> Neg for Matrix<T> {

    type Output = Matrix<T>;
    
    fn neg(mut self) -> Matrix<T> {
        let n = T::from_f64(-1.).unwrap();
        scale(&mut self, n);
        self
    }
}



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ matrix::{ Matrix }, vector::{ Vector }, matrix, vector };
    


    #[test]
    fn det() {
        //TODO
    }
    


    #[test]
    fn compose() {

        let A = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let B = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let C = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let product: Matrix<f64> = &(&A * &B) * & C;

        let product2: Matrix<f64> = compose_m!(A,B,C);

        assert_eq!(product, product2, "products should be equivalent");
    }



    #[test]
    fn from_vector() {

        let mut b: Vector<f32> = vector![1., 2., 4., 5.];

        let A : Matrix<f32> = b.into();
        
        let s = A.sum();

        assert_eq!(A.columns, 1, "from_vector - single column");

        assert_eq!(A.rows, 4, "from_vector - amount of rows equal to vector elements");

        assert_eq!(s, 12., "from_vector - sum should be 12");
    }



    #[test]
    fn negation() {

        let mut A: Matrix<f32> = Matrix::id(10);

        A = - A;
        
        let s = A.sum();

        assert_eq!(s, -10., "negation - sum should be -10");
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
    fn permutation_test() {
        let p = Matrix::<f32>::perm(4);

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
}

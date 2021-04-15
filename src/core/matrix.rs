#![feature(layout_for_ptr)]
//http://www.hpcavf.uclan.ac.uk/softwaredoc/sgi_scsl_html/sgi_html/ch03.html#ag5Plchri
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
use crate::{Number, vector, workers::Workers};
use super::{ init_const::{ init_const }, transpose::{ transpose }, id::{ id }, copy_to_f64::{ copy_to_f64 }, from_sab_f64::{ from_sab_f64 }, transfer_into_sab::{ transfer_into_sab }, into_sab::{ into_sab }, inv_diag::{ inv_diag }, inv_lower_triangular::{ inv_lower_triangular }, inv_upper_triangular::{ inv_upper_triangular }, inv::{ inv }, perm::{ perm }, rand_perm::{ rand_perm }, rank::{ rank }, partition::{ partition }, assemble::{ assemble }, cholesky::{ cholesky }, schur_complement::{ schur_complement }, lps::{ lps }, is_diagonally_dominant::{ is_diagonally_dominant }, is_identity::{ is_identity }, is_diag::{ is_diag }, is_upper_triangular::{ is_upper_triangular }, is_lower_triangular::{ is_lower_triangular }, is_permutation::{ is_permutation }, is_symmetric::{ is_symmetric }, into_basis::{ into_basis }, from_basis::{ from_basis }, eig::{ eig, eigenvectors, eig_decompose }, lu::{ block_lu, block_lu_threads_v2, lu, lu_v2 }, matrix3::Matrix3, matrix4::Matrix4, 
multiply::{ multiply_threads, mul_blocks, get_optimal_depth }, conjugate::{ conjugate },
svd::{ svd_jac1 }, qr::{qr, givens_qr, apply_Q_givens_hess, apply_Qt_givens_hess, givens_qr_upper_hessenberg, form_Qt_givens_hess, form_Q_givens_hess, apply_q_R, form_Q, form_P, house_qr, apply_q_b}, 
solve::{ solve_upper_triangular, solve, solve_lower_triangular }, utils::{eq_bound_eps, eq_bound, eq_bound_eps_v, eq_eps_f64}, vector::Vector };



/*
use Newton method
b known
compute Ax - using obtained x
compute difference d = Ax - b
transform this difference into vector in column space using:
Au = d - solve for u
next x = x - u (shifting x towards better accuracy)
*/

//debugger
//det
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



pub struct Partition <T: Number> {
    pub A11: Matrix<T>,
    pub A12: Matrix<T>,
    pub A21: Matrix<T>,
    pub A22: Matrix<T>
}



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


    //extract submatrix - takes (i,j) matrix first or last entry and dimensions
    //drop row
    //drop column
    //drop sub diagonal
    //drop cross
    pub fn extract_row(&self) {}

    pub fn extract_columns(&self) {}

    pub fn extract_rows(&self) {}

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

        rank(self)

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

        into_sab(self)

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

        let mut A = Matrix::new(rows, columns);

        let mut rng = rand::thread_rng();

        for i in 0..columns {
            for j in 0..rows {
                let value: f64 = rng.gen_range(-max, max);
                let value = ( value * 100. ).round() / 100.;
                A[[j,i]] = T::from_f64(value).unwrap();
            }
        }

        A
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



    pub fn rand_shape(max_side: usize, max:f64) -> Matrix<T> {
        
        let mut rng = rand::thread_rng();
        
        let rows = rng.gen_range(0, max_side) + 1; 

        let columns = rng.gen_range(0, max_side) + 1;

        Matrix::rand(rows, columns, max)
    }



    pub fn rand_diag(size: usize, max: f64) -> Matrix<T> {

        let mut A: Matrix<T> = Matrix::new(size,size);

        let v = Vector::rand(size as u32, max);

        A.set_diag(v);

        A
    }



    pub fn rand_sing2(size: usize, n: usize, max: f64) -> Matrix<T> {
        
        assert!(n < size, "rand_sing2: n < size");

        let mut rng = rand::thread_rng();

        let zero = T::from_f64(0.).unwrap();

        let span = size - n;

        let mut basis: Vec<Vector<T>> = Vec::new();

        let mut null: Vec<Vector<T>> = Vec::new(); 

        for i in 0..span {
            let v: Vector<T> = Vector::rand(size as u32, max);
            basis.push(v);
        }

        for i in 0..n {
            let mut v: Vector<T> = Vector::new(vec![zero; size]);

            for j in 0..basis.len() {
                let mut k = basis[j].clone();
                let s: f64 = rng.gen_range(0., 1.);
                k = k * T::from_f64(s).unwrap();
                v = v + k;
            }

            null.push(v);
        }
        
        basis.append(&mut null);

        let result = Matrix::from_basis(basis);

        let P: Matrix<T> = Matrix::rand_perm(result.columns);

        result * P
    }



    pub fn rand_sing(size: usize, max:f64) -> Matrix<T> {
        
        let mut rng = rand::thread_rng();

        let mut A = Matrix::rand(size, size, max);
        
        if size <= 1 {
            return A;
        }

        let c: usize = rng.gen_range(0, (size - 1) as u32) as usize;

        let mut basis = A.into_basis();

        println!("\n c is {} \n", c);
        
        for i in 0..A.rows {
            basis[c][i] = T::from_f64(0.).unwrap();
        }

        for i in 0..A.columns {

            if i == c {
               continue;
            }
            
            let mut v = basis[i].clone();

            let s: f64 = rng.gen_range(0., 1.);

            v = v * T::from_f64(s).unwrap();

            for j in 0..A.rows {
                basis[c][j] = basis[c][j] + v[j];
            }
        }

        Matrix::from_basis(basis)
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



    pub fn set_diag(&mut self, v:Vector<T>) {

        let c = min(self.rows, self.columns);

        assert_eq!(v.data.len(), c, "set_diag - incorrect size");

        for i in 0..c {
            self[[i, i]] = v[i];
        }
    }
    


    pub fn sum(&self) -> T {

        let mut acc = identities::zero();
    
        for i in 0..self.data.len() {
            acc = acc + self.data[i];
        }
        
        acc
    }



    pub fn trace(&self) -> T {

        let mut acc = T::from_f64(0.).unwrap();
        let d = min(self.rows, self.columns);
        
        for i in 0..d {
            acc += self[[i, i]]; 
        }

        acc
    }



    pub fn givens_theta(&self, i: usize, j: usize) -> f64 {
        let m = self[[i, j]];
        let d = self[[i - 1, j]];
        
        if d == T::from_f64(0.).unwrap() {
           return 0.;
        }

        let x: f64 = T::to_f64(&(m / d)).unwrap();
        
        let theta = x.atan();

        theta
    }



    pub fn givens(&self, i: usize, j: usize) -> Matrix<T> {
        
        let mut G: Matrix<T> = Matrix::id(self.rows);
        
        let theta = self.givens_theta(i, j);
        let s = T::from_f64( theta.sin() ).unwrap();
        let c = T::from_f64( theta.cos() ).unwrap();
        
        G[[i - 1, i - 1]] = c;
        G[[i, i - 1]] = -s;
        G[[i - 1, i]] = s;
        G[[i, i]] = c;
        
        G
    }



    /*
    inverse
    pub fn givens(&self, i: usize, j: usize) -> (Matrix<T>, Matrix<T>) {

        assert!(i > 0, "givens: i > 0");

        let mut G: Matrix<T> = Matrix::id(self.rows);
        let mut inv: Matrix<T> = Matrix::id(self.rows);
        
        let b = self[[i, j]];
        let e = self[[i + 1, j]];

        let x: f64 = T::to_f64(&(e / b)).unwrap();
        
        let phi = x.atan();
        let s = T::from_f64( phi.sin() ).unwrap();
        let c = T::from_f64( phi.cos() ).unwrap();
        
        println!("\n (f/e) is {}, (s/c) {}\n", x, (phi.sin() / phi.cos()));
        
        G[[i, j]] = c;
        G[[i + 1, j + 1]] = c;
        G[[i, j + 1]] = s;
        G[[i + 1, j]] = -s;
        
        let d = c * c + s * s;
        let k: T = c / d;
        let p: T = s / d;

        inv[[i, j]] = k;
        inv[[i + 1, j + 1]] = k;
        inv[[i, j + 1]] = -p;
        inv[[i + 1, j]] = p;
        
        (G, inv)
    }
    */



    pub fn givens2(&self, i: usize, j: usize) -> Matrix<T> {

        assert!(i > 0, "givens: i > 0");

        let mut G: Matrix<T> = Matrix::id(self.rows);

        let basis = self.into_basis();

        //cos in which dimension

        let target = &basis[i];
        let mut target2 = target.clone();
        
        target2[j] = T::from_f64(0.).unwrap();

        let l1 = target.length();
        let l2 = target2.length();
        let d: f64 = (target * &target2) / (l1 * l2);
        let phi = d.acos();
        
        let s = T::from_f64( phi.sin() ).unwrap();
        let c = T::from_f64( phi.cos() ).unwrap();
        


        println!("\n givens2: target {} \n", target);
        println!("\n givens2: target2 {} \n", target2);

        println!("\n givens2: cos {}, c {:?}, s {:?} \n", d, c, s);
        
        G[[i, j]] = c;
        G[[i + 1, j + 1]] = c;
        G[[i, j + 1]] = s;
        G[[i + 1, j]] = -s;
        
        println!("\n givens2: G is {} \n", G);

        let Gt = G.transpose();

        println!("\n givens2: Gt is {} \n", &G * target);

        println!("\n givens2: Gtt is {} \n", &Gt * target);
        
        G
    }



    pub fn project(&self, b:&Vector<T>) -> Vector<T> {

        let A = self;

        assert_eq!(A.rows, b.data.len(), "A and b dimensions should correspond");

        let mut At = A.transpose(); //TODO transpose should behave, be called in the same way for all types (same applies for remaining methods)

        let Atb: Vector<T> = &At * b;

        let AtA: Matrix<T> = &At * A;
        
        let lu = AtA.lu();

        let x = AtA.solve(&Atb, &lu).unwrap();

        A * &x
    }



    pub fn size(&self) -> usize {

        self.rows * self.columns

    } 


    
    pub fn mem_size(&self) -> usize {

        size_of::<T>() * self.rows * self.columns

    }



    pub fn is_symmetric_positive_definite() {

    }



    pub fn is_positive_definite() {

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



    pub fn round(&mut self, precision: f64) {

        let f = move |y: T| {
            let x = T::to_f64(&y).unwrap();
            let c = (2. as f64).powf(precision);
            T::from_f64((x * c).round() / c).unwrap()
        };

        self.apply(&f);
    }



    pub fn upper_hessenberg(&self) -> Matrix<T> {
        
        assert!(self.is_square(), "upper_hessenberg: A should be square");

        let zero = T::from_f64(0.).unwrap();
        
        let mut K = self.clone();
        
        if K.rows <= 2 {
           return K; 
        }
        
        for i in 0..(K.columns - 1) {
            let l = K.rows - i - 1;
            let mut x = Vector::new(vec![zero; l]);
            let mut ce = Vector::new(vec![zero; l]);

            for j in (i + 1)..K.rows {
                x[j - i - 1] = K[[j, i]];
            }

            ce[0] = T::from_f64( x.length() ).unwrap();
            
            let mut v: Vector<T> = &x - &ce;
        
            v.normalize();

            for j in i..K.columns {
                
                let mut x = Vector::new(vec![zero; v.data.len()]);
                
                for k in 0..v.data.len() {
                    x[k] = K[[k + i + 1, j]];
                }
                
                let u = &v * T::from_f64( 2. * (&v * &x) ).unwrap();
                
                for k in 0..u.data.len() {
                    K[[k + i + 1, j]] -= u[k];
                }
            }

            for j in 0..K.rows {
                
                let mut x = Vector::new(vec![zero; v.data.len()]);
                
                for k in 0..v.data.len() {
                    x[k] = K[[j, k + i + 1]];
                }
                
                let u = &v * T::from_f64( 2. * (&v * &x) ).unwrap();
                
                for k in 0..u.data.len() {
                    K[[j, k + i + 1]] -= u[k];
                }
            }
        }

        K
    }



    pub fn lower_hessenberg(&self) -> Matrix<T> {

        assert!(self.is_square(), "lower_hessenberg: A should be square");

        let zero = T::from_f64(0.).unwrap();

        let one = T::from_f64(1.).unwrap();

        let mut H = self.clone();
        
        if H.rows <= 2 {
           return H;
        }

        let f = move |x: T| {
            let y = T::to_i64(&x).unwrap();
            T::from_i64(y).unwrap()
        };

        for i in (2..(H.columns)).rev() {
            let mut x = Vector::new(vec![zero; i]);
            let mut ce = Vector::new(vec![zero; i]);

            for j in 0..i {
                x[j] = H[[j, i]];
            }
            
            ce[i - 1] = T::from_f64( x.length() ).unwrap();
            
            let mut v: Vector<T> = &x - &ce;
        
            v.normalize();

            let P = form_P(&v, H.rows, false);
            
            H = &(&P * &H) * &P;
        }

        H
    }



    pub fn is_upper_hessenberg(&self) -> bool {
        
        if !self.is_square() {
           return false; 
        }

        for i in 0..(self.columns - 2) {
            for j in (i + 2)..self.rows {
                
                if T::to_f64(&self[[j, i]]).unwrap().abs() > EPSILON as f64 {
                   return false;
                }
            }
        }

        true
    }



    pub fn is_lower_hessenberg(&self) -> bool {

        let zero = T::from_f64(0.).unwrap();

        if !self.is_square() {
           return false; 
        }

        for i in (2..self.columns).rev() {
            for j in 0..(i - 1) {

                if T::to_f64(&self[[j, i]]).unwrap().abs() > EPSILON as f64 {
                   return false;
                }
            }
        }

        true
    }



    pub fn is_diagonally_dominant(&self) -> bool {

        is_diagonally_dominant(self)

    }



    pub fn is_identity(&self) -> bool {

        is_identity(self)

    }



    pub fn is_diag(&self) -> bool {

        is_diag(self)

    }



    pub fn is_upper_triangular(&self) -> bool {
        
        is_upper_triangular(self)

    }



    pub fn is_lower_triangular(&self) -> bool {

        is_lower_triangular(self)

    }



    pub fn is_square(&self) -> bool {

        self.columns == self.rows

    }



    pub fn is_permutation(&self) -> bool {
        
        is_permutation(self)

    }



    pub fn is_symmetric(&self) -> bool {

        is_symmetric(self)

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

        let mut lu = lu_v2(self, true, true);

        lu.unwrap()

    }



    pub fn block_lu(&self) -> lu<T> {

        let mut lu = block_lu(self);

        lu.unwrap()

    }
    


    pub fn qr(&self) -> qr<T> {
        //givens_qr
        house_qr(self)

    }

    

    pub fn eigenvectors(&self, q: &Vec<f64>) -> Vec<(f64, Vector<T>)> {

        eigenvectors(self, q) 
        
    }


    
    pub fn eig(&self, precision: f64, steps: i32) -> Vec<f64> {

        eig(self, precision, steps)

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

        let mut r3: Matrix<f64> = Matrix::new(A.rows, B.columns);

        let r: Matrix<f64> = add(&r, &r3, 0); //TODO ? dim B
        
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



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ core::{lu::{block_lu_threads, block_lu_threads_v2, lu}, matrix::{ Matrix }}, matrix, vector };
    use super::{
        givens_qr_upper_hessenberg, givens_qr, form_Qt_givens_hess, form_Q_givens_hess, Matrix4,
        apply_q_b, apply_q_R, form_Q, form_P, block_lu, eq_eps_f64, Vector, P_compact, Number,
        get_optimal_depth, eq_bound_eps, eq_bound, multiply, mul_blocks, eq_bound_eps_v
    };


    
    #[test]
    fn givens_qr_upper_hessenberg_test() {

        let f = move |x: f64| {
            let c = (2. as f64).powf(32.);
            (x * c).round() / c
        };

        let size = 7;
        let max = 2.;

        let mut A: Matrix<f64> = Matrix::rand(size, size, max);
        let H = A.upper_hessenberg();

        let (R, q) = givens_qr_upper_hessenberg(H.clone(), H.columns);
        
        let Q = form_Qt_givens_hess(&q);
        let Qt = Q.transpose();
        let QR: Matrix<f64> = &Q * &R;

        //println!("\n A {}, R {}, Q {}, I {}, QR {}, diff {} \n", A, qr.R, Q, &Qt * &Q, QR, (&H - &QR));

        assert!(eq_bound_eps(&H, &QR), "H = QR");
        
        let mut R = R.clone();

        R.apply(&f);

        assert!(R.is_upper_triangular(), "R.is_upper_triangular()");
    }



    #[test]
    fn lower_hessenberg_test() {

        let test = 15;

        for i in 3..test {
            let size = i;
            let max = 50.;
            let mut A: Matrix<f64> = Matrix::rand(size, size, max);
            let mut H = A.lower_hessenberg();
            
            println!("\n H is {} \n", H);
            
            assert!(H.is_lower_hessenberg(), "H should be lower hessenberg");
        }
    }


    
    #[test]
    fn upper_hessenberg_test() {

        let test = 120;

        for i in 3..test {
            let size = 6;
            let max = 50.;
            let mut A: Matrix<f64> = Matrix::rand(size, size, max);
            let mut H = A.upper_hessenberg();
            
            println!("\n H is {} \n", H);
            assert!(H.is_upper_hessenberg(), "H should be upper hessenberg");
        }
    }



    
    



    #[test]
    fn rand_sing2_test() {

        let test = 30;

        for i in 2..test {

            let size = i;

            let max = 5.;
            
            let mut rng = rand::thread_rng();
            
            let n: u32 = rng.gen_range(0, size - 1);
            
            let A: Matrix<f64> = Matrix::rand_sing2(size as usize, n as usize, max);
    
            println!("\n A rank is {}, n is {}, size is {} \n", A.rank(), n, size);

            assert!(A.is_square(), "A is square");

            assert_eq!(A.rank() + n, size, "A.rank() + n = size");
        }
    }



    #[test]
    fn rand_sing_test() {
        
        let size = 5;

        let max = 5.;

        let A: Matrix<f64> = Matrix::rand_sing(size, max);

        assert!( A.rank() < 5, "A.rank() < 5");
    }

    

    #[test]
    fn cholesky_test() {

        let test = 20;

        for i in 2..test {
            let size = i;

            let mut A: Matrix<f64> = Matrix::rand(size, size, 1.);
    
            let f = move |x: f64| { if x < 0. { -1. * x } else { x } };
    
            let mut id: Matrix<f64> = Matrix::id(size);
    
            id = id * 100.;
    
            A.apply(&f);
    
            A = A + id;
    
            let mut At: Matrix<f64> = A.transpose();
    
            let mut A: Matrix<f64> = &At * &A;
    
            let mut L = A.cholesky();
            
            //println!("\n A ({},{}) is {} \n", A.rows, A.columns, A);
            
            if L.is_none() {
    
                println!("no Cholesky for you!");
    
            } else {
    
                let L = L.unwrap();
                
                let Lt = L.transpose();
    
                let G = &Lt * &L;
    
                assert!(eq_bound_eps(&A, &G), "A and G should be equivalent");
            }
        }
        
    }



    //#[test]
    fn block_lu_threads_test() {

        let size = 6;

        let A: Matrix<f64> = Matrix::rand(size, size, 10.);
        
        let lu = block_lu_threads_v2(&A, 1);

        let lu = lu.unwrap();

        let p: Matrix<f64> = &lu.L * &lu.U;

        let diff = &A - &p;

        println!("\n test A {} \n test L {} \n test U {} \n product {} \n diff {} \n", A, lu.L, lu.U, p, diff);

        assert!(false);
    }



    //#[test]
    fn block_lu_test() {
        
        let size = 5;

        let A: Matrix<f64> = Matrix::rand(size, size, 10.);
        
        let lu = block_lu(&A).unwrap();

        let p: Matrix<f64> = &lu.L * &lu.U;

        println!("\n A is {} \n", A);

        //println!("\n L is {} \n U is {} \n product {} \n", lu.L, lu.U, p);

        let lu2 = A.lu();

        let p2: Matrix<f64> = &lu2.L * &lu2.U;

        //println!("\n L2 is {} \n U2 is {} \n p2 is {} \n", &lu2.P * &lu2.L, lu2.U, p2);

        println!("\n p is {} \n p2 is {} \n diff is {} \n", p, p2, &p - &p2);
    }



    #[test]
    fn inv_diag() {

        let length: usize = 10;

        let v: Vector<f64> = Vector::rand(length as u32, 100.);

        let mut m: Matrix<f64> = Matrix::new(length, length);

        m.set_diag(v);

        let m_inv = m.inv_diag();

        let p: Matrix<f64> = &m_inv * &m;

        let id = Matrix::id(length);

        let equal = eq_bound_eps(&id, &p);

        assert!(equal, "inv_diag");
    }


    
    //solve upper triangular
    #[test]
    fn inv_upper_triangular() {
        let size = 10;
        let A: Matrix<f64> = Matrix::rand(size, size, 100.);
        let lu = A.lu();
        let U = lu.U; 
        let id = Matrix::id(size);

        println!("\n U is {} \n", U);

        let U_inv = U.inv_upper_triangular().unwrap();

        let p: Matrix<f64> = &U_inv * &U;
        
        let equal = eq_bound_eps(&id, &p);

        println!("\n U_inv {} \n ID is {} \n", U_inv, p);

        assert!(equal, "inv_upper_triangular");
    }


    
    #[test]
    fn inv_lower_triangular() {
        
        let size = 10;
        let A: Matrix<f64> = Matrix::rand(size, size, 100.);
        let lu = A.lu();
        let L = lu.P * lu.L; 
        let id = Matrix::id(size);

        println!("\n L is {} \n", L);

        let L_inv = L.inv_lower_triangular().unwrap();

        let p: Matrix<f64> = &L_inv * &L;
        
        let equal = eq_bound_eps(&id, &p);

        println!("\n L_inv {} \n ID is {} \n", L_inv, p);

        assert!(equal, "inv_lower_triangular");
    }



    #[test]
    fn assemble() {
        
        let A = matrix![i32,
            1, 2, 3, 6, 4, 6, 2, 5, 4;
            3, 4, 4, 4, 4, 5, 5, 5, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
        ];

        let p = A.partition(2).unwrap();

        let asm = Matrix::assemble(&p);

        assert_eq!(asm, A, "assemble 2: asm == A");

        let p = A.partition(3).unwrap();

        let asm = Matrix::assemble(&p);

        assert_eq!(asm, A, "assemble 3: asm == A");
    }



    #[test]
    fn partition() {

        let A = matrix![i32,
            1, 2;
            3, 4;
        ];

        let p = A.partition(1).unwrap();

        assert_eq!(p.A11.rows, 1, "p.A11.rows == 1");
        assert_eq!(p.A12.rows, 1, "p.A12.rows == 1");
        assert_eq!(p.A21.rows, 1, "p.A21.rows == 1");
        assert_eq!(p.A22.rows, 1, "p.A22.rows == 1");

        assert_eq!(p.A11.columns, 1, "p.A11.columns == 1");
        assert_eq!(p.A12.columns, 1, "p.A12.columns == 1");
        assert_eq!(p.A21.columns, 1, "p.A21.columns == 1");
        assert_eq!(p.A22.columns, 1, "p.A22.columns == 1");
        
        assert_eq!(p.A11[[0, 0]], 1, "p.A11[[0, 0]] == 1");
        assert_eq!(p.A12[[0, 0]], 2, "p.A12[[0, 0]] == 2");
        assert_eq!(p.A21[[0, 0]], 3, "p.A21[[0, 0]] == 3");
        assert_eq!(p.A22[[0, 0]], 4, "p.A22[[0, 0]] == 4");

        let A = matrix![i32,
            1, 2, 3, 6, 4, 6, 2, 5, 4;
            3, 4, 4, 4, 4, 5, 5, 5, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
        ];

        let p = A.partition(2).unwrap();

        println!("\n partition is \n A11 {} \n A12 {} \n A21 {} \n A22 {} \n", p.A11, p.A12, p.A21, p.A22);

        assert_eq!(p.A12.rows, 2, "p.A12.rows == 2");
        assert_eq!(p.A12.columns, 7, "p.A12.columns == 7");
    }



    fn solve9() {
        let test = 50;

        for i in 1..test {
            println!("\n solve9: working with {} \n", i);

            let size = i;

            let max = 1000.;

            let A: Matrix<f64> = Matrix::rand_shape(size, max);
            
            let b: Vector<f64> = Vector::rand(A.rows as u32, max);
            
            let lu = A.lu();
            
            let x = A.solve(&b, &lu);
            
            if x.is_some() {

                let x = x.unwrap();

                println!("\n solved with x {} \n", x);

                let Ax = &A * &x;

                for j in 0..Ax.data.len() {
                    let eq = eq_eps_f64(Ax[j], b[j]);
                    if !eq {
                        println!("\n A is {} \n", A);
                        println!("\n x is {} \n", x);
                        println!("\n b is {} \n", b);
                        println!("\n Ax is {} \n", Ax);
                        println!("\n diff is {} \n", &Ax - &b);
                    }
                    assert!(eq, "entries should be equal");
                } 
            } else {
                println!("\n no solution! projecting\n");

                let b2 = A.project(&b);

                assert!(b2 != b, "b2 and b should be different");

                let x = A.solve(&b2, &lu);

                assert!(x.is_some(), "should be able to solve for projection");

                let x = x.unwrap();
                let Ax = &A * &x;

                for j in 0..Ax.data.len() {
                    let eq = eq_eps_f64(Ax[j], b2[j]);
                    if !eq {
                        println!("\n A is {} \n", A);
                        println!("\n x is {} \n", x);
                        println!("\n b2 is {} \n", b2);
                        println!("\n Ax is {} \n", Ax);
                        println!("\n diff is {} \n", &Ax - &b2);
                    }
                    assert!(eq, "entries should be equal");
                } 
            }
        }
    }



    fn solve8() {
        let test = 50;

        for i in 1..test {
            println!("\n solve: working with {} \n", i);

            let size = i;
            let max = 1000.;
            let A: Matrix<f64> = Matrix::rand(size, size, max);
            let b: Vector<f64> = Vector::rand(size as u32, max);
            let lu = A.lu();
            let x = A.solve(&b, &lu).unwrap();
            let Ax = &A * &x;

            println!("\n solve: Ax is {} \n b is {} \n", Ax, b);

            for j in 0..Ax.data.len() {
                let eq = eq_eps_f64(Ax[j], b[j]);
                if !eq {
                    println!("\n A is {} \n", A);
                    println!("\n x is {} \n", x);
                    println!("\n b is {} \n", b);
                    println!("\n Ax is {} \n", Ax);
                    println!("\n diff is {} \n", &Ax - &b);
                }
                assert!(eq, "entries should be equal");
            }  
        }
    }



    fn solve7() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2., 1., 2., 6.;
            1., 2., 1., 1., 1., 3.;
            1., 2., 3., 1., 3., 9.; 
        ];
        let b: Vector<f64> = vector![1., 0., 2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0],-3., "x0 is -3.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
        assert_eq!(x[3], 0., "x3 is  0.");
        assert_eq!(x[4], 0., "x4 is  0.");
        assert_eq!(x[5], 0., "x5 is  0.");
    }



    fn solve6() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 2., 1.;
            1., 2., 3.;
            1., 2., 3.;
            1., 2., 3.;
        ];

        let b: Vector<f64> = vector![2., 1., 3., 3., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);
        
        assert_eq!(x[0],-2., "x0 is -2.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
    }



    fn solve5() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 2., 1.;
            1., 2., 3.;
        ];
        let b: Vector<f64> = vector![2., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} , Ax is {} \n", x, &A * &x);
        
        assert_eq!(x[0],-2., "x0 is -2.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
    }



    fn solve4() {

        let A: Matrix<f64> = Matrix::id(3);

        let b: Vector<f64> = vector![3., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0], 3., "x0 is 3.");
        assert_eq!(x[1], 1., "x1 is 1.");
        assert_eq!(x[2], 3., "x2 is 3.");
    }



    fn solve3() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 1., 1.;
            1., 2., 3.;
        ];

        let b: Vector<f64> = vector![3., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0],-1., "x0 is -1");
        assert_eq!(x[1], 2., "x1 is  2");
        assert_eq!(x[2], 0., "x2 is  0");
    }



    fn solve2() {

        let A: Matrix<f64> = matrix![f64,
            1., 5., 7.;
        ];

        let b: Vector<f64> = vector![2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();
        
        assert_eq!(x[0], 2., "x0 is 2");
        assert_eq!(x[1], 0., "x1 is 0");
        assert_eq!(x[2], 0., "x2 is 0");
    }



    fn solve1() {

        let A: Matrix<f64> = matrix![f64,
            1.;
        ];

        let b: Vector<f64> = vector![2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        assert_eq!(x[0], 2., "x is 2");
    }



    #[test]
    fn solve() {
       
       solve9();

       solve8();

       solve7();
       
       solve6();

       solve5();
    
       solve4();

       solve3();

       solve2();

       solve1();
    }



    fn lu_test10() {

        let test = 2;

        for i in 0..test {

            let max = 1000.;
        
            let max_side = 20;

            let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
        
            println!("\n lu test, iteration {}, A is ({},{}) \n", i, A.rows, A.columns);

            let mut lu = A.lu();
        
            let R: Matrix<f64> = &lu.L * &lu.U;
        
            let PL: Matrix<f64> = &lu.P * &lu.L;

            let equal = eq_bound_eps(&A, &R);

            if ! equal {
                println!(
                    "\n A is ({}, {}) {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", 
                    A, A.rows, A.columns, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P
                );
            }

            assert!(equal, "\n lu_test10 A should be equal to R \n");
        }
    }



    fn lu_test9() {
        
        let mut A: Matrix<f64> = Matrix::new(10, 10);

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        assert!(lu.U.is_upper_triangular(), "\n lu_test9 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test9 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test9 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test8() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1.;
            2.;
            3.;
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test8 A should be equal to R \n");
        
        A = A.transpose();

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test8 A transpose should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test7() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1.;
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test7 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test7 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test7 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test6() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1., 2.,  1.;
            1., 2.5, 2.;
            1., 2.9, 3.;
        ];
        
        let v = vector![0., 0., 0.];

        A.set_diag(v);

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test6 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test6 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test6 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }


    
    fn lu_test5() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2.,  1.;
            1., 2.5, 2.;
            1., 2.9, 3.;
            1., 4.,  4.; 
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        //assert!(lu.U.is_upper_triangular(), "\n lu_test5 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test5 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test5 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }


    
    fn lu_test4() {

        //TODO solve with free variables
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
        
        //assert!(lu.U.is_upper_triangular(), "\n lu_test4 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test4 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test4 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
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
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test3 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test3 PL should be lower triangular \n");
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test3 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
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
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test2 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test2 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test2 A should be equal to R \n");

        assert_eq!(lu.d.len(), 1, "\n lu_test2 d should contain 1 element \n");

        assert_eq!(lu.d[0], 1, "\n lu_test2 d should contain second col \n");
        
        assert!(id != lu.P, "\n lu_test2 P should not be identity \n");

        confirm_lu_dimensions(&A, &lu);
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
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test1 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test1 PL should be lower triangular \n");

        assert_eq!(A, R, "\n lu_test1 A should be equal to R \n");

        assert!(id != lu.P, "\n lu_test1 P should not be identity \n");

        confirm_lu_dimensions(&A, &lu);
    }
    


    fn confirm_lu_dimensions<T:Number>(A: &Matrix<T>, v: &lu<T>) {
        
        assert!(v.L.is_square(), "\n confirm_lu_dimensions L should be square \n");
        assert!(v.P.is_square(), "\n confirm_lu_dimensions P should be square \n");

        assert!(v.L.rows == A.rows, "\n L rows == A rows \n");
        assert!(v.L.columns == A.rows, "\n L columns == A rows \n");

        assert!(v.P.rows == A.rows, "\n P rows == A rows \n");
        assert!(v.P.columns == A.rows, "\n P columns == A rows \n");

        assert!(v.U.rows == A.rows, "\n L rows == A rows \n");
        assert!(v.U.columns == A.columns, "\n L columns == A rows \n");
    }



    #[test]
    fn lu_test() {
        
        lu_test1();

        lu_test2();

        lu_test3();

        lu_test4();

        lu_test5();

        lu_test6();

        lu_test7();

        lu_test8();

        lu_test9();

        lu_test10();
    }



    #[test] 
    fn rand() {

        let max = 100000.;

        let mut rng = rand::thread_rng();
        
        let value: f64 = rng.gen_range(-max, max);

        let d = ( value * 10000. ).round() / 10000.;

        println!("\n value is {} | {} \n", value, d);

    }



    #[test]
    fn rank() {
        
        let mut A1 = matrix![f64,
            5.024026017784438, 2.858902178366669, 296.2138835869165;
            7.929129970221636, 5.7210492203315795, 523.7802005055211;
            8.85257084623291, 8.95057121546899, 704.1069012250204;
        ];

        let A2 = matrix![f64,
            1.1908477166794595, 8.793086722414468, 194.77132992778556;
            3.6478484951000456, 4.858421485429982, 187.58571816777294;
            9.423462238282756, 8.321761784861303, 406.23378670237366;
        ];

        let lu = A1.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        println!("\n d1 is {:?} \n A1 is {} \n R is {} \n L is {} \n U is {} \n diff is {} \n d is {:?} \n P is {} \n \n", lu.d, A1, R, lu.L, lu.U, &A1 - &R, lu.d, lu.P);

        assert_eq!(A1.rank(), 2, "A1 - rank is 2");
        
        let lu = A2.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        println!("\n d2 is {:?} \n A2 is {} \n R is {} \n L is {} \n U is {} \n diff is {} \n d is {:?} \n P is {} \n \n", lu.d, A2, R, lu.L, lu.U, &A2 - &R, lu.d, lu.P);
        
        assert_eq!(A2.rank(), 2, "A2 - rank is 2");
        
        let test = 50;

        for i in 1..test {
            let max = 1.;

            let A: Matrix<f64> = Matrix::rand_shape(i, max);
            let At: Matrix<f64> = A.transpose();
            let AtA: Matrix<f64> = &At * &A;
            let AAt: Matrix<f64> = &A * &At;

            let rank_A = A.rank();
            let rank_AAt = AAt.rank();
            let rank_AtA = AtA.rank();

            let eq1 = rank_A == rank_AAt;
            let eq2 = rank_A == rank_AtA;

            if !eq1 {
                println!("\n rank A {}, rank AAt {} \n", rank_A, rank_AAt);
                println!("\n A ({}, {}) is {} \n AAt ({}, {}) {} \n", A.rows, A.columns, A, AAt.rows, AAt.columns, AAt);

                let luA = A.lu();
                let luAAt = AAt.lu();

                println!("\n U A is {} \n L A is {} \n d A is {:?} \n", luA.U, luA.L, luA.d);
                println!("\n U AAt is {} \n L AAt is {} \n d AAt is {:?} \n", luAAt.U, luAAt.L, luAAt.d);
            }

            if !eq2 {
                println!("\n rank A {}, rank AtA {} \n", rank_A, rank_AtA);
                println!("\n A ({}, {}) is {} \n AtA ({}, {}) {} \n", A.rows, A.columns, A, AtA.rows, AtA.columns, AtA);

                let luA = A.lu();
                let luAtA = AtA.lu();

                println!("\n U A is {} \n L A is {} \n d A is {:?} \n", luA.U, luA.L, luA.d);
                println!("\n U AtA is {} \n L AtA is {} \n d AtA is {:?} \n", luAtA.U, luAtA.L, luAtA.d);
            }
            //TODO rank AtA < rank A ??? fix
            assert!(eq1, "rank A and rank AAt should be equivalent");
            assert!(eq2, "rank A and rank AtA should be equivalent");
        }
    }



    #[test]
    fn from_basis() {

        let A = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let b = vec![
            Vector::new(vec![1., 2., 3.]),
            Vector::new(vec![2., 4., 5.]),
            Vector::new(vec![3., 7., 3.])
        ];

        let R = Matrix::from_basis(b);

        assert_eq!(A, R, "from_basis: matrices should be equal");
    }



    #[test]
    fn into_basis() {

        let A = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];

        let b = A.into_basis();

        for i in 0..b.len() {
            let col_i = &b[i];
            for j in 0..A.rows {
                assert_eq!(col_i[j], A[[j, i]], "entries should be equal");
            }
        }
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
    fn identity() {

        let id: Matrix<f64> = Matrix::id(5);
        
        assert!(id.is_diag(), "\n ID should be diagonal {} \n \n", id);
    }
    


    #[test]
    fn is_identity() {
        
        let mut A: Matrix<i32> = Matrix::id(1);
        
        assert_eq!(A.is_identity(), true, "is_identity - 1x1 id");

        A[[0, 0]] = 0;

        assert_eq!(A.is_identity(), false, "is_identity - 1x1 id");

        let mut A: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_identity(), true, "is_identity - 10x10 id");

        A[[1, 1]] = 0;

        assert_eq!(A.is_identity(), false, "is_identity - 10x10 not id");

        A[[1, 1]] = 1;
        A[[1, 0]] = 1;

        assert_eq!(A.is_identity(), false, "is_identity - 10x10 not id");
    }



    #[test]
    fn is_diagonally_dominant() {

        let mut A: Matrix<i32> = Matrix::id(10);
        let mut N: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_diagonally_dominant(), true, "is_diagonally_dominant - identity is diagonally dominant");
        
        A = A * 10;
        N = N * -2;
        
        let mut C: Matrix<i32> = Matrix::new(10, 10);

        C.init_const(1);

        A = A + N;
        A = A + C;
        
        assert_eq!(A.is_diagonally_dominant(), true, "is_diagonally_dominant - after transformation");

        A[[0, 1]] += 1;

        assert_eq!(A.is_diagonally_dominant(), false, "is_diagonally_dominant - no more");
    }



    #[test]
    fn is_diag() {

        let mut A: Matrix<i32> = Matrix::id(10);

        assert_eq!(A.is_diag(), true, "is_diag - identity is diagonal");
        
        A[[0, 9]] = 1;

        assert_eq!(A.is_diag(), false, "is_diag - A[[0, 9]] = 1, A no longer diagonal");

        A[[0, 9]] = 0;
        A[[5, 6]] = 1;

        assert_eq!(A.is_diag(), false, "is_diag - A[[5, 6]] = 1, A no longer diagonal");

        A[[5, 6]] = 0;

        assert_eq!(A.is_diag(), true, "is_diag - A is diagonal");
    }



    #[test]
    fn is_upper_triangular() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - identity is upper triangular");
        
        id[[5, 6]] = 1.;

        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - id[[6, 5]] = 1., identity is still upper triangular");

        id[[0, 9]] = 1.;
        
        assert_eq!(id.is_upper_triangular(), true, "is_upper_triangular - id[[9, 0]] = 1., identity is still upper triangular");

        id[[9, 0]] = 1.;

        assert_eq!(id.is_upper_triangular(), false, "is_upper_triangular - id[[0, 9]] = 1., identity is no longer upper triangular");
    }



    #[test]
    fn is_lower_triangular() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - identity is lower triangular");

        id[[6, 5]] = 1.;

        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - id[[6, 5]] = 1., identity is still lower triangular");

        id[[9, 0]] = 1.;
        
        assert_eq!(id.is_lower_triangular(), true, "is_lower_triangular - id[[9, 0]] = 1., identity is still lower triangular");

        id[[0, 9]] = 1.;

        assert_eq!(id.is_lower_triangular(), false, "is_lower_triangular - id[[0, 9]] = 1., identity is no longer lower triangular");
    }



    #[test]
    fn is_permutation() {

        let mut id: Matrix<f64> = Matrix::id(10);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation");

        id.exchange_rows(1, 5);
        id.exchange_rows(2, 7);
        id.exchange_rows(0, 9);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation after transform");

        id.exchange_columns(0, 9);

        assert_eq!(id.is_permutation(), true, "is_permutation - identity is permutation after col exchange");

        id[[5, 6]] = 0.1;

        assert_eq!(id.is_permutation(), false, "is_permutation - identity is no longer permutation after augmentation");
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
    fn augment_sq2n() {
        let iterations = 2;
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
    
}

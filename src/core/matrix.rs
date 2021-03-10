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
use js_sys::{Float64Array, SharedArrayBuffer};
use rand::prelude::*;
use rand::Rng;
use num_traits::{Float, Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};
use super::utils::{eq, augment_sq2n_size, augment_sq2n, init_rand, random_shape_matrix, add, subtract, multiply};
use crate::Number;



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
//strassen (how to visualize ???)



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

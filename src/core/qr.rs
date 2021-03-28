#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};

#[derive(Clone, Debug)]
pub struct qr <T: Number> {
    pub Q: Matrix<T>,
    pub R: Matrix<T>,
    pub q: Vec<Vector<T>>
}


//singular ???
//rectangular ???
pub fn qr<T: Number>(M:&Matrix<T>) -> qr<T> {

    let zero = T::from_f64(0.).unwrap();
    let one = T::from_f64(1.).unwrap();

    let mut A = M.clone();
    let mut Q: Matrix<T> = Matrix::id(M.rows);
    let mut q: Vec<Vector<T>> = Vec::new();

    for i in 0..(M.columns - 1) {

        let size = M.rows - i;
        
        let mut x = Vector::new(vec![zero; size]);
        
        for j in i..M.rows {
            x[j - i] = A[[j, i]];
        }
        
        let c = x.length();
        
        let mut ce = Vector::new(vec![zero; size]);

        ce.data[0] = one;

        ce = ce * T::from_f64(c).unwrap();
        
        let mut v: Vector<T> = &x - &ce;

        v.normalize();

        q.push(v.clone());
        
        let I: Matrix<T> = Matrix::id(M.rows);
        let u: Matrix<T> = v.clone().into();
        let ut: Matrix<T> = u.transpose();
        let uut: Matrix<T> = &u * &ut;
        let P: Matrix<T> = add(&I, &(uut * T::from_f64(-2.).unwrap()), i);

        Q = &P * &Q;
        
        for j in i..M.columns {
            
            let mut x = Vector::new(vec![zero; size]);
            
            for k in i..M.rows {
                x[k - i] = A[[k, j]];
            }

            let s = T::from_f64( 2. * (&v * &x) ).unwrap();

            let u = &v * s;

            for k in i..M.rows {
                A[[k, j]] -= u[k - i];
            }
        }
    }

    let Qt = Q.transpose();

    println!("\n Q is {} \n", Q);

    println!("\n QtQ is {} \n", &Qt * &Q);

    println!("\n R is {} \n", A);
    


    println!("\n A1 is {} \n", M);
    
    println!("\n A2 is {} \n", &Qt * &A);

    println!("\n diff is {} \n", M - &(&Qt * &A));

    qr {
        Q,
        R: A,
        q
    }
}

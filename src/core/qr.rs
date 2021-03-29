#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};

#[derive(Clone, Debug)]
pub struct qr <T: Number> {
    pub Q: Option<Matrix<T>>,
    pub Qt: Option<Matrix<T>>,
    pub R: Matrix<T>,
    pub q: Vec<Vector<T>>
}



pub fn form_P<T: Number>(v: &Vector<T>, l: usize) -> Matrix<T> {

    let size = l + 1;

    let offset = size - v.data.len();

    let I: Matrix<T> = Matrix::id(size);

    let u: Matrix<T> = v.clone().into();

    let ut: Matrix<T> = u.transpose();

    let uut: Matrix<T> = &u * &ut;

    let P: Matrix<T> = add(&I, &(uut * T::from_f64(-2.).unwrap()), offset);

    P
}



pub fn form_Q<T: Number>(q: &Vec<Vector<T>>, t: bool) -> Matrix<T> {
    
    let l = q.len();

    let mut Q: Matrix<T> = Matrix::id(l + 1);
    
    for i in 0..l {
        let v = &q[i];
        let P = form_P(v, l);

        if t {

            Q = &P * &Q;
            
        } else {

            Q = &Q * &P;
        }
    }
    
    Q
}



pub fn apply_q_b<T: Number>(q: &Vec<Vector<T>>, b: &Vector<T>, t: bool) -> Vector<T> {

    let zero = T::from_f64(0.).unwrap();

    let mut b2 = b.clone();

    let l = (q.len() - 1) as i32;

    let mut i: i32 = l;

    if t {
        i = 0;
    }

    loop {
        
        let z = i as usize;
        
        let v = &q[z];
        
        let mut x = Vector::new(vec![zero; v.data.len()]);
        
        for k in z..b2.data.len() {
            x[k - z] = b2[k];
        }

        let s = T::from_f64( 2. * (v * &x) ).unwrap();

        let u = v * s;
        
        for k in z..b2.data.len() {
            b2[k] -= u[k - z];
        }

        if t { i += 1; } else { i -= 1; }
        
        if i > l || i < 0 {
            break;
        }
    }

    b2
}



pub fn apply_q_R<T: Number>(R: &Matrix<T>, q: &Vec<Vector<T>>, t: bool) -> Matrix<T> {

    let mut QR = R.clone();

    let zero = T::from_f64(0.).unwrap();
    
    let mut i: i32 = (R.columns - 2) as i32;
    
    if t {
        i = 0;
    }

    loop {
        
        let z = i as usize;

        let size = R.rows - z;
        
        let v = &q[z];
        
        for j in z..R.columns {
            
            let mut x = Vector::new(vec![zero; size]);
            
            for k in z..R.rows {
                x[k - z] = QR[[k, j]];
            }

            let s = T::from_f64( 2. * (v * &x) ).unwrap();

            let u = v * s;
            
            for k in z..R.rows {
                QR[[k, j]] -= u[k - z];
            }
        }

        if t { i += 1; } else { i -= 1; }
        
        if i > (R.columns - 2) as i32 || i < 0 {
            break;
        }
    }

    QR
}



pub fn house_qr<T: Number>(A:&Matrix<T>) -> qr<T> {

    let zero = T::from_f64(0.).unwrap();

    let one = T::from_f64(1.).unwrap();

    let mut R = A.clone();

    let mut q: Vec<Vector<T>> = Vec::new();



    for i in 0..(A.columns - 1) {

        let size = A.rows - i;
        
        let mut x = Vector::new(vec![zero; size]);
        
        for j in i..A.rows {
            x[j - i] = R[[j, i]];
        }
        
        let c = x.length();
        
        let mut ce = Vector::new(vec![zero; size]);

        ce.data[0] = one;

        ce = ce * T::from_f64(c).unwrap();
        
        let mut v: Vector<T> = &x - &ce;

        v.normalize();

        q.push(v.clone());



        for j in i..A.columns {
            
            let mut x = Vector::new(vec![zero; size]);
            
            for k in i..A.rows {
                x[k - i] = R[[k, j]];
            }

            let s = T::from_f64( 2. * (&v * &x) ).unwrap();

            let u = &v * s;
            
            for k in i..A.rows {
                R[[k, j]] -= u[k - i];
            }
        }
    }

    qr {
        Q: None,
        Qt: None,
        R,
        q
    }
}

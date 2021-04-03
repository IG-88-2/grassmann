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



pub fn form_P<T: Number>(v: &Vector<T>, l: usize, d: bool) -> Matrix<T> {

    let size = l;

    let offset = if d { size - v.data.len() } else { 0 };

    let I: Matrix<T> = Matrix::id(size);

    let u: Matrix<T> = v.clone().into();

    let ut: Matrix<T> = u.transpose();

    let uut: Matrix<T> = &u * &ut;

    let P: Matrix<T> = add(&I, &(uut * T::from_f64(-2.).unwrap()), offset);

    P
}



pub fn form_Q<T: Number>(q: &Vec<Vector<T>>, l:usize, t: bool) -> Matrix<T> {
    
    let mut Q: Matrix<T> = Matrix::id(l);
    
    for i in 0..q.len() {
        let v = &q[i];
        let P = form_P(v, l, true);

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

    if b2.data.len() <= 1 {
       return b2; 
    }

    let l = q.len() - 1;

    let mut i = l;

    if t {
        i = 0;
    }

    loop {
        
        let v = &q[i];
        
        let mut x = Vector::new(vec![zero; v.data.len()]);
        
        for k in i..b2.data.len() {
            x[k - i] = b2[k];
        }
        
        let u = v * T::from_f64( 2. * (v * &x) ).unwrap();
        
        for k in i..b2.data.len() {
            b2[k] -= u[k - i];
        }
        
        let mut y = i as i32;

        if t { y += 1; } else { y -= 1; }
        
        if y > l as i32 || y < 0 {
            break;
        } else {
            i = y as usize;
        }
    }

    b2
}



pub fn apply_q_R<T: Number>(R: &Matrix<T>, q: &Vec<Vector<T>>, t: bool) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();

    let m = min(R.columns, R.rows) - 1;

    let mut QR = R.clone();

    if m < 1 {
        return QR;
    }
    
    let mut i = m - 1;
    
    if t {
        i = 0;
    }

    loop {
        
        let v = &q[i];
        
        for j in i..R.columns {
            
            let mut x = Vector::new(vec![zero; v.data.len()]);
            
            for k in i..R.rows {
                x[k - i] = QR[[k, j]];
            }

            let s = T::from_f64( 2. * (v * &x) ).unwrap();

            let u = v * s;
            
            for k in i..R.rows {
                QR[[k, j]] -= u[k - i];
            }
        }

        let mut y = i as i32;

        if t { y += 1; } else { y -= 1; }
        
        if y > m as i32 || y < 0 { 
            break; 
        } else {
            i = y as usize;
        }
    }

    QR
}



pub fn givens_qr<T: Number>(A:&Matrix<T>) -> qr<T> {

    let mut R = A.clone();

    let mut q: Vec<Vector<T>> = Vec::new();

    //givens(zero entry location)
    //generate givens
    //low right corner up to diagonal

    qr {
        Q: None,
        Qt: None,
        R,
        q
    }
}



pub fn house_qr<T: Number>(A:&Matrix<T>) -> qr<T> {

    let zero = T::from_f64(0.).unwrap();

    let one = T::from_f64(1.).unwrap();

    let mut R = A.clone();

    let mut q: Vec<Vector<T>> = Vec::new();

    let m = min(A.columns, A.rows) - 1;



    for i in 0..m {

        let size = A.rows - i;
        
        let mut x = Vector::new(vec![zero; size]);
        
        for j in i..A.rows {
            x[j - i] = R[[j, i]];
        }
        
        let c = x.length();
        
        let mut ce = Vector::new(vec![zero; size]);

        ce.data[0] = T::from_f64(c).unwrap();
        
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

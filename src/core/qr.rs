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



pub fn apply_q_t_givens_hess<T: Number>(m: &mut Matrix<T>, i: usize, theta: f64) {

    let zero = T::from_f64(0.).unwrap();
    let s: T = T::from_f64( theta.sin() ).unwrap();
    let c: T = T::from_f64( theta.cos() ).unwrap();

    let mut y: Vector<T> = Vector::new(vec![zero; m.columns]);
    let mut z: Vector<T> = Vector::new(vec![zero; m.columns]);
    
    for k in 0..m.columns {
        y[k] = m[[k, i + 1]];
        z[k] = m[[k, i]];
    }

    for k in 0..m.columns {
        m[[k, i]] = c * z[k] + s * y[k];
        m[[k, i + 1]] = -s * z[k] + c * y[k];
    }
}



pub fn apply_q_givens_hess<T: Number>(m: &mut Matrix<T>, i: usize, theta: f64) {

    let zero = T::from_f64(0.).unwrap();
    let s: T = T::from_f64( theta.sin() ).unwrap();
    let c: T = T::from_f64( theta.cos() ).unwrap();

    let mut y: Vector<T> = Vector::new(vec![zero; m.columns]);
    let mut z: Vector<T> = Vector::new(vec![zero; m.columns]);
    
    for k in 0..m.columns {
        y[k] = m[[i + 1, k]];
        z[k] = m[[i, k]];
    }

    for k in 0..m.columns {
        m[[i, k]] = c * z[k] + s * y[k];
        m[[i + 1, k]] = -s * z[k] + c * y[k];
    }
}



pub fn apply_Qt_givens_hess<T: Number>(A: &mut Matrix<T>, q: &Vec<(usize, f64)>) {

    let zero = T::from_f64(0.).unwrap();
    
    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_t_givens_hess(A, i, theta);
    }
}



pub fn apply_Q_givens_hess<T: Number>(A: &mut Matrix<T>, q: &Vec<(usize, f64)>) {

    let zero = T::from_f64(0.).unwrap();
    
    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_givens_hess(A, i, theta);
    }
}



pub fn form_Qt_givens_hess<T: Number>(q: &Vec<(usize, f64)>) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();
    let mut Qt: Matrix<T> = Matrix::id(q.len() + 1);

    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_t_givens_hess(&mut Qt, i, theta);
    }

    Qt
}



pub fn form_Q_givens_hess<T: Number>(q: &Vec<(usize, f64)>) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();
    let mut Q: Matrix<T> = Matrix::id(q.len() + 1);

    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_givens_hess(&mut Q, i, theta);
    }

    Q
}



pub fn givens_qr_upper_hessenberg<T: Number>(A:Matrix<T>) -> ( Matrix<T>, Vec<(usize, f64)> ) {
    
    let zero = T::from_f64(0.).unwrap();

    let mut R = A;
    
    let mut q: Vec<(usize, f64)> = Vec::new();
    
    for i in 0..(R.columns - 1) {

        let theta = R.givens_theta(i + 1, i);

        apply_q_givens_hess(&mut R, i, theta);

        let t = (i, theta);

        q.push(t);
    }
    
    (R, q)
}



pub fn apply_q_givens<T: Number>(m: &mut Matrix<T>, j: usize, theta: f64) {

    let zero = T::from_f64(0.).unwrap();
    let sin: T = T::from_f64( theta.sin() ).unwrap();
    let cos: T = T::from_f64( theta.cos() ).unwrap();
    
    let mut y: Vector<T> = Vector::new(vec![zero; m.columns]);
    let mut z: Vector<T> = Vector::new(vec![zero; m.columns]);
    
    for k in 0..m.columns {
        y[k] = m[[j, k]];
        z[k] = m[[j - 1, k]];
    }
    
    for k in 0..m.columns {
        m[[j - 1, k]] = (cos * z[k]) + (sin * y[k]);
        m[[j, k]] = (-sin * z[k]) + (cos * y[k]);
    }
}



pub fn form_Q_givens<T: Number>(A:&Matrix<T>, q: &Vec<((usize, usize), f64)>) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();
    let mut Q: Matrix<T> = Matrix::id(A.rows);

    for j in 0..q.len() {
        let (v, theta) = q[j];
        let (a, b) = v;
        apply_q_givens(&mut Q, a, theta);
    }

    Q
}



pub fn givens_qr<T: Number>(A:&Matrix<T>) -> qr<T> {

    let zero = T::from_f64(0.).unwrap();
    
    let mut A: Matrix<T> = A.clone();

    let mut q: Vec<Vector<T>> = Vec::new();

    let mut Q: Matrix<T> = Matrix::id(A.rows);

    let mut list: Vec<((usize, usize), f64)> = Vec::new();
    
    for i in (1..A.rows).rev() {

        for j in i..A.rows {

            let theta = A.givens_theta(j, j - i);
            
            let t = ((j, j - i), theta);

            list.push(t);

            apply_q_givens(&mut A, j, theta);
        }
    }

    let Q: Matrix<T> = form_Q_givens(&A, &list);
    
    let Qt = Q.transpose();

    qr {
        Q: Some(Qt),
        Qt: Some(Q),
        R: A,
        q
    }
}

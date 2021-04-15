#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};
use crate::{matrix, Number};
use super::{ 
    matrix3::Matrix3, 
    matrix4::Matrix4, 
    lu::{ 
        block_lu, 
        block_lu_threads_v2, 
        lu, 
        lu_v2
    },
    multiply::{ 
        multiply_threads,
        mul_blocks, 
        get_optimal_depth
    },
    utils::{
        eq_bound_eps_v, 
        eq_eps_f64, 
        eq_bound_eps, 
        eq_bound
    }
};



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



pub fn apply_q_t_givens_hess<T: Number>(m: &mut Matrix<T>, i: usize, theta: f64, cols: usize) {

    let zero = T::from_f64(0.).unwrap();
    let s: T = T::from_f64( theta.sin() ).unwrap();
    let c: T = T::from_f64( theta.cos() ).unwrap();

    let mut y: Vector<T> = Vector::new(vec![zero; cols]);
    let mut z: Vector<T> = Vector::new(vec![zero; cols]);
    
    for k in 0..cols {
        y[k] = m[[k, i + 1]];
        z[k] = m[[k, i]];
    }

    for k in 0..cols {
        m[[k, i]] = c * z[k] + s * y[k];
        m[[k, i + 1]] = -s * z[k] + c * y[k];
    }
}



pub fn apply_q_givens_hess<T: Number>(m: &mut Matrix<T>, i: usize, theta: f64, cols: usize, y: &mut Vector<T>, z: &mut Vector<T>) {

    let zero = T::from_f64(0.).unwrap();
    let s: T = T::from_f64( theta.sin() ).unwrap();
    let c: T = T::from_f64( theta.cos() ).unwrap();
    
    for k in 0..cols {
        y[k] = m[[i + 1, k]];
        z[k] = m[[i, k]];
    }

    for k in 0..cols {
        m[[i, k]] = c * z[k] + s * y[k];
        m[[i + 1, k]] = -s * z[k] + c * y[k];
    }
}



pub fn apply_Qt_givens_hess<T: Number>(A: &mut Matrix<T>, q: &Vec<(usize, f64)>, cols: usize) {

    let zero = T::from_f64(0.).unwrap();
    
    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_t_givens_hess(A, i, theta, cols);
    }
}



pub fn apply_Q_givens_hess<T: Number>(A: &mut Matrix<T>, q: &Vec<(usize, f64)>) {

    let zero = T::from_f64(0.).unwrap();
    let cols = A.columns;
    let mut y: Vector<T> = Vector::new(vec![zero; cols]);
    let mut z: Vector<T> = Vector::new(vec![zero; cols]);

    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_givens_hess(A, i, theta, cols, &mut y, &mut z);
    }
}



pub fn form_Qt_givens_hess<T: Number>(q: &Vec<(usize, f64)>) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();
    let mut Qt: Matrix<T> = Matrix::id(q.len() + 1);

    for j in 0..q.len() {
        let (i, theta) = q[j];
        let cols = Qt.columns;
        apply_q_t_givens_hess(&mut Qt, i, theta, cols);
    }

    Qt
}



pub fn form_Q_givens_hess<T: Number>(q: &Vec<(usize, f64)>) -> Matrix<T> {

    let zero = T::from_f64(0.).unwrap();
    let mut Q: Matrix<T> = Matrix::id(q.len() + 1);
    let cols = Q.columns;
    let mut y: Vector<T> = Vector::new(vec![zero; cols]);
    let mut z: Vector<T> = Vector::new(vec![zero; cols]);

    for j in 0..q.len() {
        let (i, theta) = q[j];
        apply_q_givens_hess(&mut Q, i, theta, cols, &mut y, &mut z);
    }

    Q
}



pub fn givens_qr_upper_hessenberg<T: Number>(A:Matrix<T>, cols: usize) -> ( Matrix<T>, Vec<(usize, f64)> ) {
    
    let zero = T::from_f64(0.).unwrap();
    let mut R = A;
    let mut q: Vec<(usize, f64)> = Vec::new();
    let mut y: Vector<T> = Vector::new(vec![zero; cols]);
    let mut z: Vector<T> = Vector::new(vec![zero; cols]);

    for i in 0..(cols - 1) {

        let theta = R.givens_theta(i + 1, i);

        apply_q_givens_hess(&mut R, i, theta, cols, &mut y, &mut z);

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



pub fn givens_qr<T: Number>(M:&Matrix<T>) -> qr<T> {

    let zero = T::from_f64(0.).unwrap();
    
    let mut R: Matrix<T> = M.clone();

    let mut q: Vec<Vector<T>> = Vec::new();

    let mut Q: Matrix<T> = Matrix::id(R.rows);

    let mut list: Vec<((usize, usize), f64)> = Vec::new();
    
    for i in (1..R.rows).rev() {

        for j in i..R.rows {

            let theta = R.givens_theta(j, j - i);

            if theta == 0. {
                continue;
            }

            let t = ((j, j - i), theta);

            list.push(t);

            apply_q_givens(&mut R, j, theta);
        }
    }

    let Q: Matrix<T> = form_Q_givens(&R, &list);
    
    let Qt = Q.transpose();

    qr {
        Q: Some(Qt),
        Qt: Some(Q),
        R,
        q
    }
}



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ core::{lu::{block_lu_threads, block_lu_threads_v2, lu}, matrix::{ Matrix }}, matrix, vector };
    use super::{
        givens_qr, apply_q_b, Matrix4, eq_eps_f64, Vector, P_compact, Number, get_optimal_depth, mul_blocks, 
        eq_bound_eps_v, eq_bound_eps, eq_bound, form_Q, form_P, apply_q_R
    };

    #[derive(PartialEq)]
    enum qr_strategy {
        house,
        givens
    }



    #[test]
    fn qr_test1() {
        
        let test = 12;
        
        let mut rng = rand::thread_rng();
        
        for i in 2..test {
            qr_test_house(i, 0);
            qr_test_house(i, 1);
            qr_test_house(i, 2);
            qr_test_house(i, 3);
        }
        
        for i in 2..test {
            //TODO
            //qr_test_givens(i, 0);
            //qr_test_givens(i, 1);
        }
    }
    


    /*
    n = 3
    matrix![f64,
        3.989662529213885, 1.3111352510863685, 0.6327979653493032, 1.2585828396533043, 0.792539040178466;
        4.805035302184294, 3.7402061208531254, 2.9985124817163875, 3.381193980206488, 2.184574576145316;
        0.10371257771584097, 3.9133316980925663, 4.030820968277014, 3.3811419789265256, 2.2285952196261243;
        1.2053697910852812, 3.882821367703303, 3.7993293346176085, 3.389835902404731, 2.2240047433478924;
        0.18136419580157326, 1.6480633417317732, 1.6725567663731522, 1.4283145902435774, 0.9401488909222216;
    ];
    n = 2
    matrix![f64,
        1.3924180940913122, 1.4694711581302726, 4.789884030718086, 2.2415031107785697;
        2.3957069426959077, 1.6044759254014513, 4.92106057298537, 2.638089492954054;
        3.8708880722626837, 0.7095293413582591, 1.184120216529141, 1.7789444693510312;
        3.4807248422705097, 1.410004021172048, 3.8392722080247497, 2.6178956559366244;
    ];
    let mut A: Matrix<f64> = matrix![f64,
        0., -2.9, -4.5, -0.03, 1.22, -3.6, 1.67, 2.79, -4.23;
        3.65, 0.72, 1.97, -4.76, 2.45, -2.89, -1.08, 4.77, 0.4;
        0.74, 0.05, -2.37, -2.46, -3.43, 3.41, -1.6, -4.53, -3.6;
        4.09, 2.62, -0.29, 1.14, -1.44, -2.16, -1.81, 4.43, -0.02;
        -4.21, 1.75, 1.72, -3.88, -1.68, -3.06, 2.12, 3.03, 4.28;
        -0.27, 4.4, -3.53, -0.7, 2.4, -4.91, 3.67, -1.61, 0.65;
        2.43, 0.91, -4.45, -1.67, 0.8, 1.24, -1.48, -4.82, 3.64;
        -4.93, -3.96, 2.48, 3.01, -2.91, 1.34, -3.18, -1.41, -2.96;
        -4.51, 4.93, 0.31, -3.97, 0.55, 4.92, 3.65, 4.84, 0.37;
    ];
    */
    fn qr_test_givens(i:usize, case:u32) {
        let mut rng = rand::thread_rng();
        
        let f = move |x: f64| {
            let c = (2. as f64).powf(32.);
            (x * c).round() / c
        };

        let size = i;
        let max = 5.;
        let mut n = 0;
        let mut A: Matrix<f64> = Matrix::rand(size, size, max);
        
        if case == 1 {
           //arbitrary singular
           n = rng.gen_range(0, size - 1);
           A = Matrix::rand_sing2(size, n, max);
        }

        println!("qr_test({}): case {} -> A({},{}), A rank {}, n {}, size {} \n", i, case, A.rows, A.columns, A.rank(), n, size);
        println!("\n A({},{}) {} \n", A.rows, A.columns, A);

        let mut qr = givens_qr(&A);
        let Q = qr.Q.unwrap();
        let Qt = qr.Qt.unwrap();
        let mut R = qr.R.clone();

        let b: Vector<f64> = Vector::rand(R.rows as u32, max);

        R.apply(&f);
        
        println!("\n Q({},{}) {} \n", Q.rows, Q.columns, Q);
        println!("\n R({},{}) {} \n", R.rows, R.columns, R);
        println!("\n R is {} \n", R);

        assert_eq!(A.rows, R.rows, "A.rows = R.rows");
        assert_eq!(A.columns, R.columns, "A.columns = R.columns");
        
        let QR: Matrix<f64> = &Q * &qr.R;
        let mut QtQ: Matrix<f64> = &Q * &Qt;
        
        QtQ.apply(&f);

        let id0 = Matrix::id(QtQ.rows);

        println!("\n QtQ {} \n", QtQ);
        println!("\n QR {} \n", QR);
        assert_eq!(QtQ, id0, "QtQ == id");
        assert!(eq_bound_eps(&A, &QR), "A = QR");

        //TODO
        //if R.is_square() {
        //   assert!(R.is_upper_triangular(), "R is upper triangular");
        //}
    }



    fn qr_test_house(i:usize, case:u32) {

        let mut rng = rand::thread_rng();
        
        let f = move |x: f64| {
            let c = (2. as f64).powf(32.);
            (x * c).round() / c
        };

        let size = i; //5;
        let max = 5.;
        let mut n = 0; //3;
        let mut A: Matrix<f64> = Matrix::rand(size, size, max);
        
        if case == 1 {
            //arbitrary singular
            n = rng.gen_range(0, size - 1);
            A = Matrix::rand_sing2(size, n, max);
        } else if case == 2 {
            //rows > columns
            let offset = rng.gen_range(1, size);
            A = Matrix::rand(size + offset, size, max);
        } else if case == 3 {
            //rows < columns
            let offset = rng.gen_range(1, size);
            A = Matrix::rand(size, size + offset, max);
        }

        println!("qr_test({}): case {} -> A({},{}), A rank {}, n {}, size {} \n", i, case, A.rows, A.columns, A.rank(), n, size);

        println!("\n A({},{}) {} \n", A.rows, A.columns, A);

        let mut qr = A.qr();

        for i in 0..qr.q.len() {
            println!("\n q({}) is {} \n", i, qr.q[i]);
        }
        
        qr.Q = Some(form_Q(&qr.q, A.rows, false));
        qr.Qt = Some(form_Q(&qr.q, A.rows, true));
            
        let Q = qr.Q.unwrap();
        let Qt = qr.Qt.unwrap();
        let mut R = qr.R.clone();

        let b: Vector<f64> = Vector::rand(R.rows as u32, max);

        R.apply(&f);
        
        println!("\n Q({},{}) {} \n", Q.rows, Q.columns, Q);
        println!("\n R({},{}) {} \n", R.rows, R.columns, R);

        println!("\n R is {} \n", R);

        assert_eq!(A.rows, R.rows, "A.rows = R.rows");
        assert_eq!(A.columns, R.columns, "A.columns = R.columns");
        
        let QR: Matrix<f64> = &Q * &qr.R;
        let Qb = &Q * &b;
        let Qtb = &Qt * &b;
        let qb = apply_q_b(&qr.q, &b, false);
        let qtb = apply_q_b(&qr.q, &b, true);
        
        assert!(eq_bound_eps_v(&qb, &Qb), "qb, Qb");
        assert!(eq_bound_eps_v(&qtb, &Qtb), "qtb, Qtb");

        let l = qr.q.len();

        let ps: Vec<Matrix<f64>> = qr.q.clone().iter_mut().map(|v| { form_P(v, R.rows, true) }).collect();

        for i in 0..ps.len() {
            let P = &ps[i];
            println!("\n P({}) is {} \n", i, P);
            assert!(P.is_symmetric(), "P should be symmetric");
        }

        let mut QtQ: Matrix<f64> = &Q * &Qt;
        
        QtQ.apply(&f);

        let id0 = Matrix::id(QtQ.rows);

        println!("\n QtQ {} \n", QtQ);
        println!("\n QR {} \n", QR);
        assert_eq!(QtQ, id0, "QtQ == id");

        let mut QR2: Matrix<f64> = apply_q_R(&qr.R, &qr.q, false);
        println!("\n QR2 {} \n", QR2);
        assert!(eq_bound_eps(&QR, &QR2), "QR = QR2");

        assert!(eq_bound_eps(&A, &QR), "A = QR");

        if R.is_square() { 
            assert!(R.is_upper_triangular(), "R is upper triangular");
        }
    }
}

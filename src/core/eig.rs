#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::{matrix, Number};
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};
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
    qr::{
        givens_qr_upper_hessenberg,
        apply_Qt_givens_hess
    },
    utils::{
        eq_bound_eps_v, 
        eq_eps_f64, 
        eq_bound_eps, 
        eq_bound
    }
};



//TODO collect eigenvectors
//TODO for symmetric matrix eigenvectors should be inside Q (verify against LU)
//TODO exceptional shifts 
//TODO double shift
pub fn eig<T: Number>(A: &Matrix<T>, precision: f64, steps: i32) -> Vec<f64> {
    
    assert!(A.rows == A.columns, "eig: A should be square");
    
    let s = 2;

    let w = true;

    let zero = T::from_f64(0.).unwrap();
    
    let mut A = A.upper_hessenberg();

    let mut q = Vec::new();

    let mut x = zero;

    let mut rows = A.rows;

    let mut cols = A.columns;



    for i in 0..steps {

        if rows <= 1 && cols <= 1 {
           let d = T::to_f64(&A[[0, 0]]).unwrap();
           q.push(d);
           break;
        }

        if i > 0 && (i % s) == 0 {
            let k = rows - 2;

            let a = T::to_f64(&A[[k, k]]).unwrap();
            let b = T::to_f64(&A[[k, k + 1]]).unwrap();
            let c = T::to_f64(&A[[k + 1, k]]).unwrap();
            let d = T::to_f64(&A[[k + 1, k + 1]]).unwrap();
            
            let u = c.abs();
            
            if u < precision {

                q.push(d);

                rows -= 1;

                cols -= 1;

            } else {

                if w {
                    let l = matrix![f64, 
                        a, b;
                        c, d;
                    ];

                    let r = Matrix::<f64>::eig2x2(&l);

                    if r.is_none() {

                        x = T::from_f64(d).unwrap();

                    } else {
                        
                        let (e1, e2) = r.unwrap();

                        let d1 = e1 - d;

                        let d2 = e2 - d;

                        if d1.abs() < d2.abs() {
                            x = T::from_f64(e1).unwrap();
                        } else {
                            x = T::from_f64(e2).unwrap();
                        }
                    }
                } else {
                    x = T::from_f64(d).unwrap();
                }
            }
        }
        
        if x != zero {
            for j in 0..rows {
                A[[j, j]] -= x;
            }
        }
        
        let (mut R, q) = givens_qr_upper_hessenberg(A, cols);

        apply_Qt_givens_hess(&mut R, &q, cols);
        
        A = R;
        
        if x != zero {
            for j in 0..rows {
                A[[j, j]] += x;
            }
        }
        
        x = zero;
    }

    q
}



pub fn eigenvectors<T: Number>(A: &Matrix<T>, q: &Vec<f64>) -> Vec<(f64, Vector<T>)> {

    let mut result: Vec<(f64, Vector<T>)> = Vec::new();

    let b: Vector<T> = Vector::zeros(A.columns);

    //println!("\n eigenvectors start b {}, rank {} \n", b, self.rank());

    for i in 0..q.len() {

        let A: Matrix<T> = A - &(Matrix::id(A.rows) * T::from_f64(q[i]).unwrap());

        //println!("\n ({}) - {} \n", i, A.rank());
        
        let lu = A.lu();

        let x = A.solve(&b, &lu);
        
        if x.is_some() {
            let t = (q[i], x.unwrap());
            result.push(t);
        }
    }

    result
}



pub fn eig_decompose<T: Number>(A: &Matrix<T>) -> Option<(Matrix<T>, Matrix<T>, Matrix<T>)> {
    
    let precision = f32::EPSILON as f64;
    let steps = 1000;
    let mut q = A.eig(precision, steps);
    let v = A.eigenvectors(&q);
    
    let ps: Vec<T> = v.iter().map(|t| { T::from_f64(t.0).unwrap() }).collect();
    let ps: Vector<T> = Vector::new(ps);
    let vs: Vec<Vector<T>> = v.iter().map(|t| { t.1.clone() }).collect();

    let mut L: Matrix<T> = Matrix::new(A.rows, A.columns);

    L.set_diag(ps);
    
    let Y = Matrix::from_basis(vs);
    let lu = Y.lu();
    let Y_inv = Y.inv(&lu);

    if Y_inv.is_none() {
       return None;
    }

    let Y_inv = Y_inv.unwrap();

    Some((Y, L, Y_inv))
}



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ core::{lu::{block_lu_threads, block_lu_threads_v2, lu}, matrix::{ Matrix }}, matrix, vector };
    use super::{
        Matrix4, eq_eps_f64, Vector, P_compact, Number, get_optimal_depth, mul_blocks, eq_bound_eps_v, eq_bound_eps, eq_bound
    };



    #[test]
    fn eigenvectors_test3() {

        let test = 20;

        for i in 2..test {
            let size = i;
            let max = 5.;
            let zero: Vector<f64> = Vector::zeros(size);
            let mut A: Matrix<f64> = Matrix::rand_diag(size, max);
            let mut a: Vector<f64> = A.extract_diag();
            let S: Matrix<f64> = Matrix::rand(size, size, max);
            let K: Matrix<f64> = A.conjugate(&S).unwrap();
    
            let R = K.eig_decompose();
    
            if R.is_some() {
                let (S, Y, S_inv) = R.unwrap();
                println!("\n K is {} \n", K);
                println!("\n S is {} \n", S);
                println!("\n Y is {} \n", Y);
                println!("\n S_inv is {} \n", S_inv);
    
                let P: Matrix<f64> = &(&S * &Y) * &S_inv;
            
                let diff = &P - &K;
                
                let bound = f32::EPSILON * 10.;

                println!("\n ({})diff {} \n", bound, diff);

                assert!(eq_bound(&K, &P, bound as f64), "\n eigenvectors_test3 K equal to P \n");
            }
        }
    }



    #[test]
    fn eigenvectors_test2() {
        
        fn round(n:&f64) -> f64 {
            let c = (2. as f64).powf(8.);
            (n * c).round() / c
        }

        let test = 20;

        for i in 2..test {
            let size = i;
            let max = 5.;
            let zero: Vector<f64> = Vector::zeros(size);
            let mut A: Matrix<f64> = Matrix::rand_diag(size, max);
            let mut a = A.extract_diag();
            let S: Matrix<f64> = Matrix::rand(size, size, max);
            let K: Matrix<f64> = A.conjugate(&S).unwrap();
            let precision = f32::EPSILON as f64;
            let steps = 1000;
            let mut q = K.eig(precision, steps);
            let v = K.eigenvectors(&q);
    
            for i in 0..v.len() {
                let (e, k) = &v[i];
                let id: Matrix<f64> = Matrix::id(size);
                let M: Matrix<f64> = id * *e;
                let mut y: Vector<f64> = &(&K - &M) * k;
                y.apply(round);
                println!("\n y {} \n", y);
                assert!(eq_bound_eps_v(&zero, &y));
            }
        }
    }


    
    #[test]
    fn eigenvectors_test1() {

        let test = 20;

        for i in 2..test {

            let size = i;
        
            let max = 5.0;
            
            let mut K: Matrix<f64> = Matrix::rand_diag(size, max);
            
            let precision = f32::EPSILON as f64;
            
            let steps = 1000;
            
            let q = K.eig(precision, steps);
            
            let v = K.eigenvectors(&q);
    
            let vs: Vec<Vector<f64>> = v.iter().rev().map(|t| { t.1.clone() }).collect();
    
            let y = Matrix::from_basis(vs);
    
            let id: Matrix<f64> = Matrix::id(size);
    
            assert_eq!(y, id, "y == id");
        }
    }



    #[test]
    fn eig4() {

        let m = Matrix4::rotation(0.4, 0.3, 0.2);
        let K: Matrix<f64> = m.into();
        let K: Matrix<f64> = K.lps(3);

        let precision = f32::EPSILON as f64;
        let steps = 10000;
        let mut q = K.eig(precision, steps);

        //why eigenvalues are not empty ?

        println!("\n K is {} \n result 3 is {:?} \n", K, q);

        //assert!(false);
    }



    #[test]
    fn eig3() {
        
        let K = matrix![f64,
            0., 0., 0., 1.;
            1., 0., 0., 0.;
            0., 1., 0., 0.;
            0., 0., 1., 0.;
        ];
        let precision = f32::EPSILON as f64;
        let steps = 1000;
        let mut q = K.eig(precision, steps);

        println!("\n result 2 is {:?} \n", q);
    }



    #[test]
    fn eig2() {
        
        let K = matrix![f64,
            0., 1.;
            1., 0.;
        ];
        let precision = f32::EPSILON as f64;
        let steps = 1000;
        let mut q = K.eig(precision, steps);

        println!("\n result is {:?} \n", q);
    }



    #[test]
    fn eig1() {
        
        let test = 20;

        for i in 2..test {
            eig_test(i);
        }
    }



    fn eig_test(i: usize) {

        println!("\n eig test {} \n", i);

        let f = move |x: &mut f64| -> f64 {
            let c = (2. as f64).powf(12.);
            (*x * c).round() / c
        };

        let size = i;
        let max = 5.;
        let mut A: Matrix<f64> = Matrix::rand_diag(size, max);
        let mut a = A.extract_diag();
        let S: Matrix<f64> = Matrix::rand(size, size, max);
        let K: Matrix<f64> = A.conjugate(&S).unwrap();
        
        let precision = f32::EPSILON as f64;
        let steps = 1000;
        let mut q = K.eig(precision, steps);
        let mut c = Vector::new(q);
        
        c.data = c.data.iter_mut().map(&f).collect();
        a.data = a.data.iter_mut().map(&f).collect();

        c.data.sort_by(|a, b| b.partial_cmp(a).unwrap());
        a.data.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        //println!("\n expected {:?} \n", a.data);
        println!("\n result {:?} \n", c.data);
        println!("\n equal {} \n", a == c);

        assert_eq!(a, c, "a == c");
    }
}

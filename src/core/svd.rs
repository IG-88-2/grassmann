#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};
use super::{ 
    lu::{ block_lu, block_lu_threads_v2, lu, lu_v2 }, 
    matrix3::Matrix3, 
    matrix4::Matrix4, 
    multiply::{ multiply_threads, strassen, mul_blocks, get_optimal_depth, decompose_blocks },
    utils::{eq_bound_eps_v, eq_eps_f64, eq_bound_eps, eq_bound}
};
/*
let d = aj_aj - ai_ai;
if d == 0. {
    continue;
}
let z: f64 = ai_aj / d;
let t = ((2. * z).atan()) / 2.;
let c = t.cos();
let s = t.sin();

let mut R: Matrix<T> = Matrix::id(A.columns);
R[[i, i]] = c;
R[[j, i]] = s;
R[[j, j]] = c;
R[[i, j]] = -s;
V = &V * &R;
A = &A * &R;
*/
//AV = UE
//A = UEVt
//AtA = VE^2Vt
//VtV = UtU = I

//TODO parallelize



pub fn svd_jac1_form_UE<T: Number>(A: Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    
    let zero = T::from_f64(0.).unwrap();

    let mut cols = A.into_basis();

    let mut e: Vec<T> = Vec::new();
    
    for p in 0..cols.len() {
        let next = &cols[p];
        
        let len = T::from_f64( next.length() ).unwrap();
        
        if len == zero {
           continue;
        }
        
        e.push(len);
        
        cols[p] = next / len;
    }

    let U = Matrix::from_basis(cols);

    let d = e.len();

    let mut E: Matrix<T> = Matrix::new(d, d); 

    let diag = Vector::new(e);
    
    E.set_diag(diag);

    (U, E)
}



pub fn svd_jac1<T: Number>(mut A: Matrix<T>, eps: f64) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        
    let zero = T::from_f64(0.).unwrap();

    let mut V: Matrix<T> = Matrix::id(A.columns);
    let mut vi: Vector<T> = Vector::new(vec![zero; V.rows]);
    let mut vj: Vector<T> = Vector::new(vec![zero; V.rows]);
    let mut ai: Vector<T> = Vector::new(vec![zero; A.rows]); 
    let mut aj: Vector<T> = Vector::new(vec![zero; A.rows]);

    loop {

        let mut ctr = 0;

        for i in 0..A.columns {

            let c = A.extract_column(i);

            for j in (i + 1)..A.columns {
                
                let x = A.extract_column(j);

                let ai_aj: f64 = &c * &x;
                let aj_aj: f64 = &c * &c;
                let ai_ai: f64 = &x * &x;
                
                if ai_aj.abs() < eps {
                   continue;
                }

                let w = (aj_aj - ai_ai) / (2. * ai_aj);
                let t = w.signum() / (w.abs() + (1. + w.powf(2.)).sqrt()); 
                let c = 1. / (1. + t.powf(2.)).sqrt();
                let s = c * t;
                let c = T::from_f64(c).unwrap();
                let s = T::from_f64(s).unwrap();
                
                for k in 0..V.rows {
                    vi[k] = V[[k, i]];
                    vj[k] = V[[k, j]];
                }

                for k in 0..A.rows {
                    ai[k] = A[[k, i]];
                    aj[k] = A[[k, j]];
                }
                
                for k in 0..V.rows {
                    V[[k, i]] = c * vi[k] + s * vj[k];
                    V[[k, j]] = -s * vi[k] + c * vj[k];
                }

                for k in 0..A.rows {
                    A[[k, i]] = c * ai[k] + s * aj[k];
                    A[[k, j]] = -s * ai[k] + c * aj[k];
                }

                ctr += 1;
            }
        }

        if ctr == 0 {
           break;
        }
    }
    
    let (U, E) = svd_jac1_form_UE(A);
    
    let Vt = V.transpose();
    
    (U, E, Vt)
}



mod tests {
    use num::Integer;
    use rand::Rng;
    use std::{ f32::EPSILON as EP, f64::EPSILON, f64::consts::PI };
    use crate::{ core::{lu::{block_lu_threads, block_lu_threads_v2, lu}, matrix::{ Matrix }}, matrix, vector };
    use super::{
        Matrix4, eq_eps_f64, Vector, P_compact, Number, get_optimal_depth, mul_blocks, 
        strassen, decompose_blocks, eq_bound_eps_v, eq_bound_eps, eq_bound
    };



    //#[test]
    fn svd_jac1_test2() {

        let size = 150;

        let max = 5.;

        let mut A: Matrix<f64> = Matrix::rand(size, size, max);

        let eps = (f32::EPSILON) as f64;

        svd_jac1_test0(&A, eps);
    }



    #[test]
    fn svd_jac1_test1() {

        let test = 20;
        
        //square
        for i in 2..test {
            let size = i;
            let max = 5.;
            let mut A: Matrix<f64> = Matrix::rand(size, size, max);
            let eps = (f32::EPSILON) as f64;
            svd_jac1_test0(&A, eps);
        }
        
        //singular
        for i in 2..test {
            let mut rng = rand::thread_rng();
            let size = i;
            let max = 5.;
            let n = rng.gen_range(0, size - 1);
            let A: Matrix<f64> = Matrix::rand_sing2(size, n, max);
            let eps = (f32::EPSILON) as f64;
            svd_jac1_test0(&A, eps);
        }
        
        //rows > columns
        for i in 2..test {
            let mut rng = rand::thread_rng();
            let size = i;
            let max = 5.;
            let n = rng.gen_range(0, size - 1);
            let mut A: Matrix<f64> = Matrix::rand(size + n, size, max);
            let eps = (f32::EPSILON) as f64;
            svd_jac1_test0(&A, eps);
        }

        //columns > rows
        for i in 2..test {
            let mut rng = rand::thread_rng();
            let size = i;
            let max = 5.;
            let n = rng.gen_range(0, size - 1);
            let mut A: Matrix<f64> = Matrix::rand(size + n, size, max);
            let eps = (f32::EPSILON) as f64;
            svd_jac1_test0(&A, eps);
        }
    }


    
    fn svd_jac1_test0(A: &Matrix<f64>, eps:f64) {

        let r = A.rank() as usize;
        let At: Matrix<f64> = A.transpose();
        let mut AtA: Matrix<f64> = &At * A;
        let d = 7.;
        let (mut U, E, Vt) = A.clone().svd(eps);
        
        assert_eq!(U.rows, A.rows, "U rows == A rows");
        assert_eq!(U.columns, A.columns, "U columns == A columns");
        assert_eq!(Vt.rows, A.columns, "Vt rows == A columns");
        assert_eq!(Vt.columns, A.columns, "Vt columns == A columns");
        assert_eq!(E.rows, A.columns, "E rows == A columns");
        assert_eq!(E.columns, A.columns, "E columns == A columns");
        assert!(E.is_square(), "E is square");
        assert!(Vt.is_square(), "Vt is square");

        println!("\n E is {} \n", E);
        
        //VtV = UtU = I
        println!("\n U is {} \n", U);
        let Ut = U.transpose();
        let mut UtU: Matrix<f64> = &Ut * &U;
        //UtU.round(4.);
        println!("\n UtU is {} \n", UtU);
        let V = Vt.transpose();
        println!("\n V is {} \n", V);
        let mut VtV: Matrix<f64> = &Vt * &V;
        //VtV.round(4.);
        println!("\n VtV is {} \n", VtV);

        let mut UtUr = UtU.clone();
        UtUr.round(d);
        let mut VtVr = VtV.clone();
        VtVr.round(d);

        println!("\n UtUr is {} \n", UtUr);
        println!("\n VtVr is {} \n", VtVr);

        if !(A.is_square() && r < A.rows) {
           assert!(eq_bound_eps(&UtUr, &Matrix::id(UtU.rows)), "UtU == I");
        }
        
        assert!(eq_bound_eps(&VtVr, &Matrix::id(VtV.rows)), "VtV == I");

        let AV: Matrix<f64> = A * &V;
        let UE: Matrix<f64> = &U * &E;
        assert!(eq_bound_eps(&AV, &UE), "AV == UE");

        let E2: Matrix<f64> = &E * &E;
        let mut VE2Vt: Matrix<f64> = &V * &(&E2 * &Vt);

        println!("\n AtA {} \n", AtA);
        println!("\n VE^2Vt {} \n", VE2Vt);

        let mut AtAr = AtA.clone();
        let mut VE2Vtr = VE2Vt.clone();

        AtAr.round(d);
        VE2Vtr.round(d);

        assert!(eq_bound_eps(&AtAr, &VE2Vtr), "AtA == VE^2Vt");
        
        let EVt: Matrix<f64> = &E * &Vt;
        let P: Matrix<f64> = &U * &EVt;

        println!("\n P is {} \n", P);
        assert!(eq_bound_eps(&A, &P), "A == P");
        
        let mut e_diag = E.extract_diag();

        e_diag.data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut eig = AtA.eig(eps, 200);

        eig = eig.iter().map(|x| { if *x <= (f32::EPSILON as f64) { 0. } else { x.sqrt() } }).collect();
        
        eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut ata_eig = Vector::new(eig.clone());

        e_diag.data = e_diag.data.iter_mut().map(|x| { 
            let c = (2. as f64).powf(d);
            (*x * c).round() / c
        }).collect();

        ata_eig.data = ata_eig.data.iter_mut().map(|x| {  
            let c = (2. as f64).powf(d);
            (*x * c).round() / c
        }).collect();

        println!("\n e_diag ({}) is {} \n", e_diag.data.len(), e_diag);

        println!("\n AtA eig ({}) sqrt is {} \n", ata_eig.data.len(), ata_eig);

        if ata_eig.data.len() == e_diag.data.len() {
            assert!(eq_bound_eps_v(&e_diag, &ata_eig), "e_diag == ata_eig");
        }
    }

}
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, P_compact};



#[derive(Clone, Debug)]
pub struct lu <T: Number> {
    pub L: Matrix<T>,
    pub U: Matrix<T>,
    pub P: Matrix<T>,
    pub d: Vec<u32>
}

//TODO 
//matrix type pre-checks (symmetric, positive definite, etc -> optional equilibration + partial pivoting)
//TODO
//equilibration logic should be more sophisticated
//read https://cs.stanford.edu/people/paulliu/files/cs517-project.pdf
//i can equilibrate both columns and rows E(LU)Q
//TODO
//? bitwise operations for performance

fn equilibrate <T: Number> (U:&mut Matrix<T>) -> (Matrix<T>, Matrix<T>) {

    let mut E: Matrix<T> = Matrix::id(U.rows);

    let mut Q: Matrix<T> = Matrix::id(U.columns);

    let zero = T::from_f64(0.).unwrap();

    let one = T::from_f64(1.).unwrap();



    for i in 0..U.rows {
        let thresh: T = T::from_f64((2. as f64).powf(16.)).unwrap();
        let mut l = zero;
        let mut b = zero;
        let mut s = one;

        for j in 0..U.columns {
            let v = U[[i,j]];
            let n = v * v;

            if v.abs() > b.abs() {
                b = v; 
            }

            if v.abs() < s.abs() {
                s = v; 
            }
            
            l += n;
        }

        if (b - s).abs() > thresh {
            continue;
        }

        let v = T::to_f64(&l).unwrap().sqrt();

        l = T::from_f64(v).unwrap();

        if T::to_f32(&l).unwrap() < f32::EPSILON {
            for j in 0..U.columns {
                //U[[i,j]] /= l;
            }
            //E[[i,i]] = l;
        }
    }

    (E, Q)
}



pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let steps = min(A.rows, A.columns);
    let mut P: P_compact<T> = P_compact::new(A.rows);
    let mut U: Matrix<T> = A.clone();
    let mut L: Matrix<T> = Matrix::new(A.rows, A.rows);
    let mut d: Vec<u32> = Vec::new();
    let mut row = 0;
    let mut col = 0;

    let (E, Q) = equilibrate(&mut U);
    
    for i in 0..steps {
        let mut k = row;
        let mut p = U[[row, col]];
        
        for j in (row + 1)..U.rows {
            let c = U[[j, col]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }  
        }

        //println!("evaluate next {} | {}", T::to_f32(&p).unwrap(), f32::EPSILON);

        if T::to_f32(&p).unwrap().abs() < f32::EPSILON {

            d.push(i as u32);

            col += 1;

            continue;

        } else {

            if k != row {
                P.exchange_rows(row, k);
                U.exchange_rows(row, k);
            }
        }
        
        for j in row..U.rows {
            let h = T::to_i32(&P.map[j]).unwrap() as usize;
            let e: T = U[[j, col]];
            let c: T = e / p;
            
            L[[h, row]] = c;

            if j == row {
               continue;
            }
            
            for t in col..U.columns {
                U[[j,t]] = U[[j,t]] - U[[row,t]] * c;
            }
        }
        
        //println!("\n next step {} \n L {} \n U {} \n ", i + 1, L, U);

        row += 1;
        col += 1;
    }
    
    lu {
        P: P.into_p(),
        L: &E * &L,
        U,
        d
    }
}

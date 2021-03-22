#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, P_compact, Partition};

/*
    pre-pivoting
    partition r - n, r - m 
    A11 A12
    A21 A22

    L11 0       U11 U12
    L21 L22       0 U22
    
    L11 * U11     L11 * U12  
    L21 * U11     L21 * U12 + L22 * U22

    A11 = L11 * U11;
    A21 = L21 * U11 ---> A21 * U11inv = L21;
    A12 = L11 * U12 ---> L11inv * A12 = U12;
    L11, U11, L21, U12
    A22 = L21 * U12 + L22 * U22
    A22 = A21 * U11inv * L11inv * A12 + L22 * U22
    I = U11inv * L11inv * L11 * U11, L11 * U11 = A11 -> A11inv = U11inv * L11inv
    A22 = A21 * A11inv * A12 + L22 * U22
    SC = A21 * A11inv * A12
    A22 - SC = L22 * U22

    task A = L11 * U11
    1*task L11 -> L11inv
    1*task U11 -> U11inv
    task A11inv = U11inv * L11inv
    task SC = A21 * A11inv * A12
    task L21 = A21 * U11inv
    task U12 = L11inv * A12
*/

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



pub fn block_lu_threads<T: Number>(A: &Matrix<T>, r: usize) -> Option<lu<T>>   {

    if r >= A.rows {
        println!("use shortcut \n");
        let lu = A.lu();
        return Some(lu);
    }

    let mut L: Matrix<T> = Matrix::new(A.rows, A.rows); 
    let mut U: Matrix<T> = Matrix::new(A.rows, A.columns); 
    let mut P: Matrix<T> = Matrix::id(L.rows);

    let mut p = A.partition(r);

    let steps = A.rows / r;
    
    println!("\n A ({},{}) - steps {} \n", A.rows, A.columns, steps);



    for i in 0..steps {
        
        let offset = r * i;

        let asmb = Matrix::assemble(&p);

        println!("\n A' ({},{}) is {} \n", asmb.rows, asmb.columns, asmb);
        println!("\n step {}, offset {} \n", i, offset);
        println!("\n current L {} \n", L);
        println!("\n current U {} \n", U);

        let A11_lu = lu_v2(&p.A11, false, false).unwrap();
        let L11: Matrix<T> = A11_lu.L;
        let U11: Matrix<T> = A11_lu.U;

        
        let L11_inv: Matrix<T> = L11.inv_lower_triangular().unwrap(); 
        let U11_inv: Matrix<T> = U11.inv_upper_triangular().unwrap();
        let L21: Matrix<T> = &p.A21 * &U11_inv;
        let U12: Matrix<T> = &L11_inv * &p.A12;
        let SC: Matrix<T> = Matrix::schur_complement(&p).unwrap();
        
        let A22: Matrix<T> = p.A22.clone();

        let next: Matrix<T> = A22 - SC;

        /*
        L11 0       U11 U12
        L21 L22       0 U22
        */

        for j in 0..L11.rows {
            for k in 0..L11.columns {
                L[[j + offset, k + offset]] = L11[[j, k]];
            }
        }

        for j in 0..L21.rows {
            for k in 0..L21.columns {
                L[[j + offset + r, k + offset]] = L21[[j, k]];
            }
        }

        for j in 0..U11.rows {
            for k in 0..U11.columns {
                U[[j + offset, k + offset]] = U11[[j, k]];
            }
        }

        for j in 0..U12.rows {
            for k in 0..U12.columns {
                U[[j + offset, k + offset + r]] = U12[[j, k]];
            }
        }
        
        if r <= next.columns && r <= next.rows {

            p = next.partition(r);

        } else {

            let lu = next.lu();
            
            for j in 0..lu.L.rows {
                for k in 0..lu.L.columns {
                    L[[j + offset, k + offset]] = lu.L[[j, k]];
                }
            }

            for j in 0..lu.U.rows {
                for k in 0..lu.U.columns {
                    U[[j + offset, k + offset]] = lu.U[[j, k]];
                }
            }

            println!("finale");
        }
    }
    
    let result = lu {
        L,
        U,
        P,
        d: Vec::new()
    };

    return Some(result);
}



pub fn block_lu<T: Number>(A: &Matrix<T>) -> Option<lu<T>> {
    
    let r = 2;

    let steps = min(A.rows, A.columns);
    
    if steps <= r {
        
        return lu_v2(A, false, false);

    } else {

        let p = A.partition(r);
        
        let lu_A11 = lu_v2(&p.A11, false, false);

        if lu_A11.is_none() {
           return None; 
        }

        let lu_A11 = lu_A11.unwrap();

        let L11 = lu_A11.L.clone();
        let U11 = lu_A11.U.clone();
        
        let L11_inv = L11.inv_lower_triangular().unwrap();
        let U11_inv = U11.inv_upper_triangular().unwrap();

        let A11_inv: Matrix<T> = &U11_inv * &L11_inv; //p.A11.inv(&lu_A11).unwrap();
        
        let L21: Matrix<T> = &p.A21 * &U11_inv;
        let U12: Matrix<T> = &L11_inv * &p.A12;

        let SC: Matrix<T> = &(&p.A21 * &A11_inv) * &p.A12;

        let A22: Matrix<T> = p.A22 - SC;
        
        let next = block_lu(&A22).unwrap();

        /*
        L11 0       U11 U12
        L21 L22       0 U22
        */
        
        let L22 = next.L;
        let U22 = next.U;
        let L12 = Matrix::new(U12.rows, U12.columns);
        let U21 = Matrix::new(L22.rows, L22.columns);

        let L_p = Partition {
            A11: L11,
            A21: L21,
            A22: L22,
            A12: L12
        };
        
        let U_p = Partition {
            A11: U11,
            A21: U21,
            A22: U22,
            A12: U12
        };

        let L: Matrix<T> = Matrix::assemble(&L_p);
        let U: Matrix<T> = Matrix::assemble(&U_p);
        let P: Matrix<T> = Matrix::id(L.rows);

        let result = lu {
            L,
            U,
            P,
            d: Vec::new()
        };

        Some(result)
    }
}



pub fn lu_v2<T: Number>(A: &Matrix<T>, eq:bool, pp: bool) -> Option<lu<T>> {

    let steps = min(A.rows, A.columns);
    let mut P: P_compact<T> = P_compact::new(A.rows);
    let mut U: Matrix<T> = A.clone();
    let mut L: Matrix<T> = Matrix::new(A.rows, A.rows);
    let mut d: Vec<u32> = Vec::new();
    let mut row = 0;
    let mut col = 0;
    let mut E: Option<Matrix<T>> = None;
    let mut Q: Option<Matrix<T>> = None;

    if eq {
        let r = equilibrate(&mut U);
        E = Some(r.0); 
        Q = Some(r.1);
    }

    for i in 0..steps {
        let mut k = row;
        let mut p = U[[row, col]];
        
        if pp {
            for j in (row + 1)..U.rows {
                let c = U[[j, col]];
                if c.abs() > p.abs() {
                    p = c;
                    k = j;
                }  
            }
        }

        //floating point arithmetic issue, which epsilon should i choose ? 
        let eps = f32::EPSILON * 10.; //f32::EPSILON

        //println!("evaluate next {} | {}", T::to_f32(&p).unwrap(), eps);

        if T::to_f32(&p).unwrap().abs() < eps {

            if pp {

                d.push(i as u32);

                col += 1;

                continue;

            } else {

                return None;

            }
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
    
    if eq {
       L = &(E.unwrap()) * &L;
    }

    Some(
        lu {
            P: P.into_p(),
            L,
            U,
            d
        }
    )
}

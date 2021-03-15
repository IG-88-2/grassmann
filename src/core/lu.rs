use std::{cmp::min, collections::{HashMap, HashSet}};
use crate::Number;
use super::matrix::Matrix;



#[derive(Clone, Debug)]
pub struct lu <T: Number> {
    pub L: Matrix<T>,
    pub U: Matrix<T>,
    pub P: Matrix<T>, 

    /*
    E: Matrix<T>,
    Q: Matrix<T>
    */
}

/*
pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);
    
    let mut L: Matrix<T> = Matrix::id(size);
    let mut U: Matrix<T> = A.clone();
    let mut P: Matrix<T> = Matrix::id(size);
    
    for i in 0..(U.rows - 1) {
        let mut k = i;
        let mut p = U[[i, i]];
        
        for j in (i + 1)..U.rows {
            let c = U[[j, i]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }    
        }
        
        if k != i {
            P.exchange_rows(i, k);
            U.exchange_rows(i, k);
        }

        //goal get rid of Ln, inject c in M according to P rule
        let mut Ln: Matrix<T> = Matrix::id(size);
        let mut Pn: Matrix<T> = Matrix::id(size);

        for j in (i + 1)..U.rows {
            let e: T = U[[j, i]];
            let c: T = e / p;
            
            Ln[[j, i]] = c;

            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[i,t]] * c;
            }
        }
        
        //Pn.exchange_rows(i, k); //P Ln - action on rows
        
        //L = &L * &(&Pn * &Ln);
        L.exchange_columns(i, k);

        L = &L * &(&Pn * &Ln);
        
        println!("\n step {} L is {} \n Ln is {} \n", i, L, Ln);
    }

    lu {
        P,
        L,
        U
    }
}
*/

pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);
    let zero = T::from_i32(0).unwrap();
    let mut L: Matrix<T> = Matrix::id(size);
    let mut Ltest: Matrix<T> = Matrix::id(size);

    let mut U: Matrix<T> = A.clone();
    let mut P: Matrix<T> = Matrix::id(size);
    let mut Q: Matrix<T> = Matrix::id(size);
    let mut Ptest: Matrix<T> = Matrix::id(size);

    
    for i in 0..U.rows {
        let mut k = i;
        let mut p = U[[i, i]];
        
        for j in (i + 1)..U.rows {
            let c = U[[j, i]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }    
        }
        
        if k != i {
            P.exchange_rows(i, k);
            U.exchange_rows(i, k);
        }
        
        let mut Ln: Matrix<T> = Matrix::id(size);
        
        //Matrix::init_const(&mut Ln, 0.);

        for j in (i + 1)..U.rows {
            let e: T = U[[j, i]];
            let c: T = e / p;
            
            Ln[[j, i]] = c;

            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[i,t]] * c;
            }
        }
        
        //L.exchange_columns(i, k); //L P
        
        let mut Pk: Matrix<T> = Matrix::id(size);
        Pk.exchange_rows(i, k);
        Ptest = &Ptest * &Pk;
        let mut tmp: Matrix<T> = &Ptest * &Ln.clone();

        //Ln Q
        //let tmp = &Ln * &Q;
        //L = &L * &P;


        for j in 0..U.rows {
            Ltest[[j, i]] = tmp[[j, i]]; //Ln[[j, i]];
        }

        L = &(&L * &Pk) * &Ln;
        
        //easy step produce from Ln column vector of the same for as being embedded into L
        
        println!("\n step {} \n L is {} \n Ln {} \n Ltest {}", i + 1, L, tmp, Ltest);
    }

    //L = &L * &P;

    lu {
        P,
        L: Ltest,
        U
    }
}



pub fn lu_v1<T: Number>(A: &Matrix<T>) -> lu<T> {

    let mut h: HashSet<usize> = HashSet::new();
    let size = min(A.columns, A.rows);
    let zero = T::from_i32(0).unwrap();

    let mut U: Matrix<T> = A.clone();
    let mut L: Matrix<T> = Matrix::id(size);
    let mut P: Matrix<T> = Matrix::id(size);
    let mut Q: Matrix<T> = Matrix::id(size);

    println!("\n U start is {} \n", U);

    for i in 0..(U.rows - 1) {
        let mut k = i;
        let mut p = U[[i, i]];
        
        for j in (i + 1)..U.rows {
            if h.contains(&j) {
                continue;
            }
            let c = U[[j, i]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }    
        }
        
        if p == zero {
            for j in ((i + 1)..U.columns).rev() {
                p = U[[i, j]];

                if p != zero {
                    U.exchange_columns(i, j);
                    break;
                }
            }

            if p == zero {
                break;
            }
        }
        
        if k != i {
            P.exchange_rows(i, k);
        }
        
        h.insert(k);

        for j in 0..U.rows {
            if h.contains(&j) {
                continue;
            }

            let e: T = U[[j, i]];
            let c: T = e / p;
            
            L[[j, k]] = c;
            
            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[k,t]] * c;
            }
        }
        
        println!("\n U next is {} \n", U);
    }

    lu {
        P,
        L,
        U
    }
}

/*
pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);
    
    let mut L: Matrix<T> = Matrix::id(size);
    let mut U: Matrix<T> = A.clone();
    let mut M: Matrix<T> = Matrix::id(size);
    let mut P: Matrix<T> = Matrix::id(size);
    //let mut Q: Matrix<T> = Matrix::id(size);
    //let mut E: Matrix<T> = Matrix::id(A.rows);
    
    for i in 0..(U.rows - 1) {
        let mut k = i;
        let mut p = U[[i, i]];
        
        for j in (i + 1)..U.rows {
            let c = U[[j, i]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }    
        }
        
        if k != i {
            P.exchange_rows(i, k);
            U.exchange_rows(i, k);
        }
        
        /*
        if p == zero {
            singular = true;
            for j in ((i + 1)..U.columns).rev() {
                p = U[[i, j]];
                
                for q in (i + 1)..U.rows {
                    let c = U[[q, j]];
                    if c.abs() > p.abs() {
                        p = c;
                        k = q;
                    }
                }

                if p != zero {
                    
                    U.exchange_columns(i, j);
                    Q.exchange_columns(i, j);

                    U.exchange_rows(i, k);
                    P.exchange_rows(i, k);
                    s *= -1;
                    break;
                }
            }

            if p == zero {
                break;
            }
        }
        */

        //goal get rid of Ln, inject c in M according to P rule
        let mut Ln: Matrix<T> = Matrix::id(size);

        for j in (i + 1)..U.rows {
            let e: T = U[[j, i]];
            let c: T = e / p;
            
            Ln[[j, i]] = c;

            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[i,t]] * c;
            }
        }


        
        Ln.exchange_rows(i, k);
        
        M = &M * &(Ln); //add rearranged column

        println!("\n ({}) M is {} \n Ln is {} \n", i, M, Ln);
    }

    lu {
        //E, 
        //P, 
        //Q, 
        P,
        L: M,
        U
    }
}
*/

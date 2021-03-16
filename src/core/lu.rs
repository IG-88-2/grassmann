use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, P_compact};



#[derive(Clone, Debug)]
pub struct lu <T: Number> {
    pub L: Matrix<T>,
    pub U: Matrix<T>,
    pub P: Matrix<T>, 
    pub Q: Matrix<T>
    /*
    E: Matrix<T>, //divide by biggest number in the row
    */
}

//Q
//rows > columns
//columns > rows
//singular
//initial equilibration 
pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);
    let mut P: P_compact<T> = P_compact::new(size);
    //apply Q immediately, apply Q after ?
    let mut Q: P_compact<T> = P_compact::new(A.columns);
    let mut U: Matrix<T> = A.clone();
    let mut L: Matrix<T> = Matrix::new(size, size);

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
        
        //this columns i bad
        if T::to_f32(&p).unwrap() == 0. {

            //find first usable column, keep track of flawed columns

            //println!("\n step {} \n matrix is singular \n A {} \n U {} \n L {} \n", i + 1, A, U, L);
            
            // skip column, reorder later ?

            for j in ((i + 1)..U.columns).rev() {
                p = U[[i, j]];
                
                for q in (i + 1)..U.rows {
                    let c = U[[q, j]];
                    if c.abs() > p.abs() {
                        p = c;
                        k = q;
                    }
                }

                if T::to_f32(&p).unwrap() != 0. {
                    Q.exchange_rows(i, j);
                    //U.exchange_columns(i, j);
                    
                    P.exchange_rows(i, k);
                    U.exchange_rows(i, k);
                    break;
                }
            }

            println!("\n step {} \n U {} \n L {} \n", i + 1, U, L);

            if T::to_f32(&p).unwrap() == 0. {
                println!("\n step {} \n done for singular matrix \n A {} \n U {} \n L {} \n", i + 1, A, U, L);
                break;
            }



        } else {

            if k != i {
                P.exchange_rows(i, k);
                U.exchange_rows(i, k);
            }
        }
        
        for j in i..U.rows {
            let h = T::to_i32(&P.map[j]).unwrap() as usize;
            let e: T = U[[j, i]];
            let c: T = e / p;

            L[[h, i]] = c;

            if j == i {
               continue;
            }
            
            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[i,t]] * c;
            }
        }
    }
    
    lu {
        P: P.into_p(),
        L,
        U,
        Q: P.into_p()
    }
}

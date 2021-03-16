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
//edge cases - [1] [1,2,3] [1,2,3]T [0,0,0], all ones etc

pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);
    let mut P: P_compact<T> = P_compact::new(size);
    let mut Q: P_compact<T> = P_compact::new(A.columns);
    let mut U: Matrix<T> = A.clone();
    let mut L: Matrix<T> = Matrix::new(size, size);
    let mut d: Vec<u32> = Vec::new();
    let mut row = 0;
    let mut col = 0;
    
    for i in 0..U.rows {
        let mut k = row;
        let mut p = U[[row, col]];
        
        for j in (row + 1)..U.rows {
            let c = U[[j, col]];
            if c.abs() > p.abs() {
                p = c;
                k = j;
            }    
        }
        
        if T::to_f32(&p).unwrap() == 0. {
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
        
        println!("\n next step {} \n L {} \n U {} \n ", i + 1, L, U);

        row += 1;
        col += 1;
    }
    
    //wrong how can i keep them upper triangular ?

    let mut last = A.columns - 1;

    for i in 0..d.len() {
        let next = d[i] as usize;
        Q.exchange_rows(next, last);
        last -= 1;
    }

    println!("\n done Q is {} \n", Q.into_p());
    
    lu {
        P: P.into_p(),
        Q: Q.into_p(),
        L,
        U
    }
}

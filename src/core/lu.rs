use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use crate::Number;
use super::matrix::{Matrix, P_compact};



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

//Q
//rows > columns
//columns > rows
//singular
//initial equilibration 
pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {

    let size = min(A.columns, A.rows);

    let mut P: P_compact<T> = P_compact::new(size);
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
        
        if T::to_f32(&p).unwrap() == 0. {
           println!("\n step {} \n matrix is singular \n A {} \n U {} \n L {} \n", i + 1, A, U, L);
        }

        if k != i {
            P.exchange_rows(i, k);
            U.exchange_rows(i, k);
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
        U
    }
}



/*
pub fn lu_v2<T: Number>(A: &Matrix<T>) -> lu<T> {
    //robust and stable for square matrices
    //singular
    //col > row
    //row > col
    let mut permutation: HashMap<usize, usize> = HashMap::new();

    let mut p2: HashMap<usize, usize> = HashMap::new();

    let zero = T::from_i32(0).unwrap();
    let size = min(A.columns, A.rows);

    for i in 0..size {  
        permutation.insert(i, i);
    }

    let mut L: Matrix<T> = Matrix::id(size);
    let mut L_test: Matrix<T> = Matrix::id(size);

    let mut U: Matrix<T> = A.clone();
    let mut P: Matrix<T> = Matrix::id(size);
    
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
        
        //below epsilon
        if p == zero {
           break;
        }

        if k != i {
            let index1 = *permutation.get(&i).unwrap();
            let index2 = *permutation.get(&k).unwrap();

            permutation.insert(i, index2);
            permutation.insert(k, index1);

            //p2.insert(i, k);
            p2.insert(k, i);
            
            P.exchange_rows(i, k);
            U.exchange_rows(i, k);
        }
        
        let mut Ln: Matrix<T> = Matrix::id(size);
        
        for j in (i + 1)..U.rows {
            let e: T = U[[j, i]];
            let c: T = e / p;
            let idx = p2.get(&j);//.unwrap();
            //keep track of permutation transposed ?
            let index = *permutation.get(&j).unwrap();
            
            let row = index;

            let index2 = *permutation.get(&index).unwrap();

            //let row = index2;

            Ln[[j, i]] = c; //remove

            //current row is j
            //get hash
            //
            if idx.is_some() {
                let n = *idx.unwrap();
                L[[n, i]] = c;
            } else {
                L[[j, i]] = c;
            }

            for t in i..U.columns {
                U[[j,t]] = U[[j,t]] - U[[i,t]] * c;
            }
        }
        
        //remove
        let P_t = P.transpose();
        let tmp: Matrix<T> = &P_t * &Ln;
        for j in 0..U.rows {
            L_test[[j, i]] = tmp[[j, i]];
        }


        let mut P_verify: Matrix<T> = Matrix::id(size); 
        
        Matrix::init_const(&mut P_verify, 0.);

        for i in 0..size {
            let col = *permutation.get(&i).unwrap();
            P_verify[[i,col]] = T::from_i16(1).unwrap();
        }
        
        //println!("\n step {} \n L is {} \n Ln {} \n L {}", i + 1, L, tmp, L);
        println!("\n step {} \n L good is {} \n L bad {} \n P is {} \n Pver is {}", i + 1, L_test, L, P, P_verify);
    }
    
    lu {
        P,
        L: L_test,
        U
    }
}
*/

/*
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
*/
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

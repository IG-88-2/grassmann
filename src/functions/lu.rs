use std::cmp::min;

use crate::{Number, Partition, core::matrix::Matrix};

use super::{p_compact::P_compact, utils::eq_bound_eps};


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



pub fn block_lu_threads_v2<T: Number>(A: &Matrix<T>, r: usize) -> Option<lu<T>>   {

    let mut L: Matrix<T> = Matrix::new(A.rows, A.rows); 
    let mut U: Matrix<T> = Matrix::new(A.rows, A.columns); 
    let mut P: Matrix<T> = Matrix::id(L.rows);

    let mut A = A.clone();
    let mut ctr = 0;

    loop {
        
        //println!("\n ({}) A is {} \n", ctr, A);

        let p = A.partition(r);

        if p.is_some() {
            let p = p.unwrap();
            let c = Matrix::schur_complement(&p);

            if c.is_none() {
               return None;
            }
    
            let c = c.unwrap();
            
            let lu = lu_v2(&p.A11, false, false);
            
            if lu.is_none() {
               return None;
            }

            let lu = lu.unwrap();
            
            let L11: Matrix<T> = lu.L;
            let U11: Matrix<T> = lu.U;
            
            let L11_inv = L11.inv_lower_triangular();

            if L11_inv.is_none() {
               return None;
            }

            let L11_inv = L11_inv.unwrap();
            let U11_inv = U11.inv_upper_triangular();

            if U11_inv.is_none() {
               return None;
            }

            let U11_inv = U11_inv.unwrap();

            let s = ctr * r;
            
            let L11: Matrix<T> = L11;
            let U11: Matrix<T> = U11;
            let L21: Matrix<T> = &p.A21 * &U11_inv;
            let U12: Matrix<T> = &L11_inv * &p.A12;

            //println!("\n |{}| L11 {} \n U11 {} \n", ctr, L11, U11);

            //println!("\n |{}| L21 {} \n U12 {} \n", ctr, L21, U12);
            
            /*
            L11 0       U11 U12
            L21 L22       0 U22
            
            A11 A12 A13    L11 0   0       U11 U12 U13
            A21 A22 A23    L21 L22 0       0   U22 U23
            A31 A32 A33    L31 L32 L33     0   0   U33    
            */
            
            for j in 0..L11.rows {
                for k in 0..L11.columns {
                    L[[s + j, s + k]] = L11[[j, k]];
                }
            }

            for j in 0..U11.rows {
                for k in 0..U11.columns {
                    U[[s + j, s + k]] = U11[[j, k]];
                }
            }
            
            for j in 0..L21.rows {
                for k in 0..L21.columns {
                    L[[s + j + r, s + k]] = L21[[j, k]];
                }
            }
            
            for j in 0..U12.rows {
                for k in 0..U12.columns {
                    U[[s + j, s + k + r]] = U12[[j, k]];
                }
            }
            
            A = p.A22 - c;
            
        } else {
            
            let lu = lu_v2(&A, false, false);
            
            if lu.is_none() {
                return None;
            }

            let lu = lu.unwrap();

            let s = ctr * r;

            for j in 0..lu.L.rows {
                for k in 0..lu.L.columns {
                    L[[s + j, s + k]] = lu.L[[j, k]];
                }
            }

            for j in 0..lu.U.rows {
                for k in 0..lu.U.columns {
                    U[[s + j, s + k]] = lu.U[[j, k]];
                }
            }

            break;
        }

        ctr += 1;

        println!("\n ({}) U is {} \n", ctr, U);

        println!("\n ({}) L is {} \n", ctr, L);
    }

    println!("\n ({}) U is {} \n", ctr, U);

    println!("\n ({}) L is {} \n", ctr, L);

    let result = lu {
        L,
        U,
        P,
        d: Vec::new()
    };

    Some(result)
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

    let mut p = A.partition(r).unwrap();

    let steps = A.rows / r;
    
    println!("\n A ({},{}) - steps {} \n", A.rows, A.columns, steps);



    for i in 0..steps {
        
        let offset = r * i;
        
        //println!("\n A' ({},{}) is {} \n", asmb.rows, asmb.columns, asmb);
        //println!("\n step {}, offset {} \n", i, offset);
        //println!("\n current L {} \n", L);
        //println!("\n current U {} \n", U);

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
        
        if r < next.columns && r < next.rows {
            println!("\n ({}) next is {} \n", i, next);
            p = next.partition(r).unwrap();
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

        let p = A.partition(r).unwrap();
        
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


//TODO separate for rectangular and square
//TODO solve multiple rhs at once with solve
//TODO write to A
//TODO statically permutable matrix
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



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };
    use super::{eq_bound_eps, lu, block_lu_threads_v2, block_lu};


    fn lu_test10() {

        let test = 2;

        for i in 0..test {

            let max = 1000.;
        
            let max_side = 20;

            let mut A: Matrix<f64> = Matrix::rand_shape(max_side, max);
        
            println!("\n lu test, iteration {}, A is ({},{}) \n", i, A.rows, A.columns);

            let mut lu = A.lu();
        
            let R: Matrix<f64> = &lu.L * &lu.U;
        
            let PL: Matrix<f64> = &lu.P * &lu.L;

            let equal = eq_bound_eps(&A, &R);

            if ! equal {
                println!(
                    "\n A is ({}, {}) {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", 
                    A, A.rows, A.columns, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P
                );
            }

            assert!(equal, "\n lu_test10 A should be equal to R \n");
        }
    }



    fn lu_test9() {
        
        let mut A: Matrix<f64> = Matrix::new(10, 10);

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        assert!(lu.U.is_upper_triangular(), "\n lu_test9 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test9 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test9 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test8() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1.;
            2.;
            3.;
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test8 A should be equal to R \n");
        
        A = A.transpose();

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test8 A transpose should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test7() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1.;
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test7 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test7 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test7 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test6() {
        
        let mut A: Matrix<f64> = matrix![f64,
            1., 2.,  1.;
            1., 2.5, 2.;
            1., 2.9, 3.;
        ];
        
        let v = vector![0., 0., 0.];

        A.set_diag(v);

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test6 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test6 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test6 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }


    
    fn lu_test5() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2.,  1.;
            1., 2.5, 2.;
            1., 2.9, 3.;
            1., 4.,  4.; 
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        //assert!(lu.U.is_upper_triangular(), "\n lu_test5 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test5 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test5 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }


    
    fn lu_test4() {

        //TODO solve with free variables
        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 1., 1., 1.;
            1., 2.5, 2., 2., 12.;
            1., 2.9, 3., 1., 7.;
            1., 4., 4., 2., 3.; 
        ];

        let mut lu = A.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        //assert!(lu.U.is_upper_triangular(), "\n lu_test4 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test4 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test4 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test3() {
        /*
        let mut A: Matrix<f32> = matrix![f32,
            1.3968362, -0.97569525, -5.018955, 0.7136311;
            -4.6254315, 9.305554, 2.5439813, 1.9787005;
            -3.233496, -4.881222, -3.2327516, 3.0223584;
            -1.1067164, -4.347563, -8.04766, 1.6895233;
        ];
        */
        let mut A: Matrix<f64> = matrix![f64,
            1.3968362, -0.0009525, -5.018955, 23352352350.7136311; //-0.00009525 TODO
            -4.6254315, 9.305554, 2.5439813, 1234234234.9787005;
            -3.233496, -4.881222, -3.2327516, 3534534534.0223584;
            -1.1067164, -445645645645.347563, -8.04766, 1634634634.6895233;
        ];
    
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test3 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test3 PL should be lower triangular \n");
        
        assert!(eq_bound_eps(&A, &R), "\n lu_test3 A should be equal to R \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test2() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 1., 1.;
            1., 2., 2., 2.;
            1., 2., 3., 1.;
            1., 2., 4., 2.;
        ];
        
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test2 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test2 PL should be lower triangular \n");

        assert!(eq_bound_eps(&A, &R), "\n lu_test2 A should be equal to R \n");

        assert_eq!(lu.d.len(), 1, "\n lu_test2 d should contain 1 element \n");

        assert_eq!(lu.d[0], 1, "\n lu_test2 d should contain second col \n");
        
        assert!(id != lu.P, "\n lu_test2 P should not be identity \n");

        confirm_lu_dimensions(&A, &lu);
    }



    fn lu_test1() {

        let mut A: Matrix<f64> = matrix![f64,
            1., 2., 3.;
            2., 4., 7.;
            3., 5., 3.;
        ];
        
        let mut lu = A.lu();
        
        let R: Matrix<f64> = &lu.L * &lu.U;

        let PL: Matrix<f64> = &lu.P * &lu.L;

        let id: Matrix<f64> = Matrix::id(lu.P.rows);

        println!("\n A is {} \n R is {} \n L is {} \n U is {} \n PL is {} \n diff is {} \n d is {:?} \n P is {} \n", A, R, lu.L, lu.U, PL, &A - &R, lu.d, lu.P);
        
        assert!(lu.U.is_upper_triangular(), "\n lu_test1 U should be upper triangular \n");

        assert!(PL.is_lower_triangular(), "\n lu_test1 PL should be lower triangular \n");

        assert_eq!(A, R, "\n lu_test1 A should be equal to R \n");

        assert!(id != lu.P, "\n lu_test1 P should not be identity \n");

        confirm_lu_dimensions(&A, &lu);
    }
    


    fn confirm_lu_dimensions<T:Number>(A: &Matrix<T>, v: &lu<T>) {
        
        assert!(v.L.is_square(), "\n confirm_lu_dimensions L should be square \n");
        assert!(v.P.is_square(), "\n confirm_lu_dimensions P should be square \n");

        assert!(v.L.rows == A.rows, "\n L rows == A rows \n");
        assert!(v.L.columns == A.rows, "\n L columns == A rows \n");

        assert!(v.P.rows == A.rows, "\n P rows == A rows \n");
        assert!(v.P.columns == A.rows, "\n P columns == A rows \n");

        assert!(v.U.rows == A.rows, "\n L rows == A rows \n");
        assert!(v.U.columns == A.columns, "\n L columns == A rows \n");
    }



    #[test]
    fn lu_test() {
        
        lu_test1();

        lu_test2();

        lu_test3();

        lu_test4();

        lu_test5();

        lu_test6();

        lu_test7();

        lu_test8();

        lu_test9();

        lu_test10();
    }

    

    //#[test]
    fn block_lu_threads_test() {

        let size = 6;

        let A: Matrix<f64> = Matrix::rand(size, size, 10.);
        
        let lu = block_lu_threads_v2(&A, 1);

        let lu = lu.unwrap();

        let p: Matrix<f64> = &lu.L * &lu.U;

        let diff = &A - &p;

        println!("\n test A {} \n test L {} \n test U {} \n product {} \n diff {} \n", A, lu.L, lu.U, p, diff);

        assert!(false);
    }



    //#[test]
    fn block_lu_test() {
        
        let size = 5;

        let A: Matrix<f64> = Matrix::rand(size, size, 10.);
        
        let lu = block_lu(&A).unwrap();

        let p: Matrix<f64> = &lu.L * &lu.U;

        println!("\n A is {} \n", A);

        //println!("\n L is {} \n U is {} \n product {} \n", lu.L, lu.U, p);

        let lu2 = A.lu();

        let p2: Matrix<f64> = &lu2.L * &lu2.U;

        //println!("\n L2 is {} \n U2 is {} \n p2 is {} \n", &lu2.P * &lu2.L, lu2.U, p2);

        println!("\n p is {} \n p2 is {} \n diff is {} \n", p, p2, &p - &p2);
    }
}
#![allow(dead_code, warnings)]
use std::{f32::EPSILON, mem::size_of, ops::Index};
use crate::Number;
use super::matrix::Matrix;
use js_sys::{Float64Array, SharedArrayBuffer};
use num_traits::identities;
use rand::prelude::*;
use rand::Rng;
use wasm_bindgen::{JsCast, prelude::*};



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn decompose_blocks<T:Number>(A: Matrix<T>) -> (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>) {

    assert_eq!(A.rows, A.columns, "matrix should be square");
    assert_eq!((A.rows as f32).log2().fract(), 0., "matrix should be pow 2");

    let r = A.rows / 2;
    let c = A.columns / 2;
    
    let mut A11 = Matrix::new(r, c);
    let mut A12 = Matrix::new(r, c);
    let mut A21 = Matrix::new(r, c);
    let mut A22 = Matrix::new(r, c);
    
    for i in 0..r {
        for j in 0..c {
            A11[[i, j]] = A[[i, j]];
            A12[[i, j]] = A[[i, c + j]];
            A21[[i, j]] = A[[r + i, j]];
            A22[[i, j]] = A[[r + i, c + j]];
        }
    }


    (A11, A12, A21, A22)
}



pub fn recombine_blocks<T:Number>(A11: Matrix<T>, A12: Matrix<T>, A21: Matrix<T>, A22: Matrix<T>) -> Matrix<T> {

    assert_eq!(A11.rows, A11.columns, "A11 matrix should be square");
    assert_eq!((A11.rows as f32).log2().fract(), 0., "A11 matrix should be pow 2");
    
    assert_eq!(A11.rows, A12.rows, "A11 should have rows equivalent to A12");
    assert_eq!(A11.columns, A12.columns, "A11 should have columns equivalent to A12");
    
    assert_eq!(A11.rows, A21.rows, "A11 should have rows equivalent to A21");
    assert_eq!(A11.columns, A21.columns, "A11 should have columns equivalent to A21");

    assert_eq!(A11.rows, A22.rows, "A11 should have rows equivalent to A22");
    assert_eq!(A11.columns, A22.columns, "A11 should have columns equivalent to A22");

    let rows = A11.rows;
    let columns = A11.columns;
    let r = rows * 2;
    let c = columns * 2;

    let mut A = Matrix::new(r, c);

    for i in 0..rows {
        for j in 0..columns {
            A[[i,j]] = A11[[i,j]];
            A[[i,j + columns]] = A12[[i,j]];
            A[[i + rows,j]] = A21[[i,j]];
            A[[i + rows,j + columns]] = A22[[i,j]];
        }
    } 

    A
}



pub fn augment_sq2n_size<T:Number>(A: &Matrix<T>) -> usize {

    if A.rows==A.columns && (A.rows as f32).log2().fract() == 0. {
       return A.rows;
    }

    let mut side = std::cmp::max(A.rows, A.columns);
    let l: f64 = (side as f64).log2();
    if l.fract() != 0. {
        side = (2. as f64).powf(l.ceil()) as usize;
    }
    
    side
}



pub fn augment_sq2n<T:Number>(A: Matrix<T>) -> Matrix<T> {

    if A.rows <= 1 && A.columns <= 1 {
       return A;
    }
    
    if A.rows==A.columns && (A.rows as f32).log2().fract() == 0. {
        return A;
    }
    
    let mut side = std::cmp::max(A.rows, A.columns);
    let l: f64 = (side as f64).log2();
    
    if l.fract() != 0. {
        side = (2. as f64).powf(l.ceil()) as usize;
    }
    
    let mut m: Matrix<T> = Matrix::new(side, side);
    m = m + A;
    m
}



pub fn init_rand<T: Number>(A: &mut Matrix<T>) {
    let max = 683.45;
    let mut rng = rand::thread_rng();
    for i in 0..A.columns {
        for j in 0..A.rows {
            let value: f64 = rng.gen();
            A[[j,i]] = T::from_f64(value).unwrap();
        }
    }
}



pub fn init_const<T: Number>(A: &mut Matrix<T>) {
    let max = 683.;
    let c = 6.6; //rng.gen_range((0.)..max);
    for i in 0..A.columns {
        for j in 0..A.rows {
            A[[j,i]] = T::from_f64(c).unwrap();
        }
    }
}



pub fn random_shape_matrix(max: usize) -> Matrix<f64> {
    let A_rows = 22; //rng.gen_range(0..max) + 1; 
    let A_columns = 12; //rng.gen_range(0..max) + 1;
    let mut eA: Matrix<f64> = Matrix::new(A_rows, A_columns);

    init_rand(&mut eA);
    eA
} 



pub fn division_level<T: Number>(A: &Matrix<T>, optimal_block_size: usize, threads: usize) -> usize {

    let mut s = optimal_block_size;

    if (s as i32) == -1 {
        s = 10000;
    }
    
    unsafe {
        log(&format!("\n threads {} \n", threads));
    }

    let limit = false;

    let total_size = A.size();

    let mut blocks = 1;

    let x = total_size / optimal_block_size;

    if x < 1 {

        blocks = 1;

    } else {

        let c = if x > threads && limit { threads } else { x };

        let n = (c as f64).log(4.).ceil();

        blocks = (4. as f64).powf(n) as usize;
    }

    blocks
}



#[wasm_bindgen]
pub fn ml_thread(
    sab:&SharedArrayBuffer,

    a_rows:usize,
    a_columns:usize,
    b_rows:usize,
    b_columns:usize,
    
    ax0:usize,ay0:usize,
    ax1:usize,ay1:usize,
    bx0:usize,by0:usize,
    bx1:usize,by1:usize
) {
    let s = size_of::<f64>();
    let mut sa = a_rows * a_columns * s;
    let mut sb = b_rows * b_columns * s;
    
    let mut ra = Float64Array::new_with_byte_offset(&sab, 0);
    let mut rb = Float64Array::new_with_byte_offset(&sab, sa as u32);
    let mut rc = Float64Array::new_with_byte_offset(&sab, (sa + sb) as u32);
    
    let mut ca = 0.;
    let mut cb = 0.;
    
    for m in ay0..ay1 {
        for n in ax0..ax1 {
            ca += ra.get_index((m * a_columns + n) as u32);
        }
    }

    if ca == 0. {
        return;
    }
    
    for m in by0..by1 {
        for n in bx0..bx1 {
            cb += rb.get_index((m * b_columns + n) as u32);
        }
    }

    if cb == 0. {
        return;
    }
    
    for m in ay0..ay1 {
        for n in bx0..bx1 {
            for p in ax0..ax1 {
                let v = rc.get_index((m * b_columns + n) as u32) + 
                        ra.get_index((m * a_columns + p) as u32) * 
                        rb.get_index((p * b_columns + n) as u32);

                rc.set_index((m * b_columns + n) as u32, v);
            }
        }
    }
}



pub fn multiply_blocks_threads(
    a: &mut Matrix<f64>, 
    b: &mut Matrix<f64>, 
    optimal_block_size: usize,
    threads: usize
) -> (
    usize,
    Matrix<f64>,
    Matrix<f64>,
    Vec<[usize; 8]>
) {
    
    let mut tasks: Vec<[usize; 8]> = Vec::new();

    let s1 = augment_sq2n_size(&a);

    let s2 = augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<f64> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<f64> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    //println!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks);

    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;
    
    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {
            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);
                
                let t: [usize; 8] = [
                    ax0,ay0,
                    ax1,ay1,
                    bx0,by0,
                    bx1,by1
                ];

                tasks.push(t);
            }
        }
    }

    (
        blocks,
        A,
        B,
        tasks
    )
}



pub fn multiply_blocks<T: Number>(
    a: &mut Matrix<T>, 
    b: &mut Matrix<T>, 
    optimal_block_size: usize,
    discard_zero_blocks: bool,
    threads: usize
) -> Matrix<T> {

    let s1 = augment_sq2n_size(&a);

    let s2 = augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<T> = Matrix::new(s,s); 

    A = &A + a;

    let mut B: Matrix<T> = Matrix::new(s,s); 

    B = &B + b;
    
    let blocks = division_level(&A, optimal_block_size, threads);
    
    //println!("A:({},{}), B:({},{}), size {}, computing with {} blocks", A.rows, A.columns, B.rows, B.columns, A.size(), blocks);

    let block_size = A.size() / blocks;

    let y = (block_size as f64).sqrt() as usize;

    let mut acc: Matrix<T> = Matrix::new(A.rows, B.columns);

    for i in (0..A.rows).step_by(y) {
        for j in (0..B.columns).step_by(y) {

            let mut v:Vec<Vec<f64>> = Vec::new();

            for k in (0..A.columns).step_by(y) {
                let ax0 = k;
                let ax1 = (k + y);
                let ay0 = i;
                let ay1 = (i + y);
                
                let bx0 = j;
                let bx1 = (j + y);
                let by0 = k;
                let by1 = (k + y);
                
                //this is it

                if discard_zero_blocks {
                    let mut sa = T::zero();
                    let mut sb = T::zero();

                    for m in ay0..ay1 {
                        for n in ax0..ax1 {
                            sa += A[[m,n]];
                        }
                    }

                    if sa == T::zero() {
                        continue;
                    }
                    
                    for m in by0..by1 {
                        for n in bx0..bx1 {
                            sb += B[[m,n]];
                        }
                    }

                    if sb == T::zero() {
                        continue;
                    }
                }
                
                for m in ay0..ay1 {
                    for n in bx0..bx1 {
                        for p in ax0..ax1 {
                            acc[[m,n]] += A[[m,p]] * B[[p,n]];    
                        }
                    }
                }
            }
        }
    }

    acc
}



pub fn get_optimal_depth<T: Number> (A: &Matrix<T>, optimal_element_size: usize) -> usize {

    assert_eq!(A.rows, A.columns, "A should be square matrix");

    let p = (optimal_element_size as f64).log(4.).ceil();

    let optimal_element_size = (4. as f64).powf(p) as usize;

    let size = A.rows * A.columns;

    if size < 64 {
        return 0;
    }
    
    let chunks = size / optimal_element_size;

    if chunks < 2 {
        return 4;
    }
    
    (chunks as f64).log(4.).ceil() as usize
}



pub fn transpose <T:Number>(m: &Matrix<T>) -> Matrix<T> {

    let mut t = Matrix::new(m.columns, m.rows);

    for i in 0..m.rows {
        for j in 0..m.columns {
            t[[j,i]] = m[[i,j]];
        }
    }

    t
}



pub fn add <T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    //TODO mistake fix
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");

    let mut C: Matrix<T> = Matrix::new(A.rows, A.columns);
    
    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i,j]] = A[[i,j]] + B[[i,j]];
        }
    }
    
    C
}



pub fn subtract<T: Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {
    
    assert!(A.rows >= B.rows, "rows do not match");

    assert!(A.columns >= B.columns, "columns do not match");

    let mut C: Matrix<T> = Matrix::new(A.rows, A.columns);
    
    for i in 0..B.rows {
        for j in 0..B.columns {
            C[[i,j]] = A[[i,j]] - B[[i,j]];
        }
    }
    
    C
}



fn sum <T: Number>(len: usize, data: impl Index<usize, Output = T>) -> T {
    
    let mut acc = identities::zero();
    
    for i in 0..len {
        acc = acc + data[i];
    }
    
    acc
}



fn identity <T: Number>(size: usize) -> Matrix<T> {

    let mut data: Vec<T> = Vec::with_capacity(size*size);
    
    for i in 0..size {
        data[(size * i) + i] = T::from_i32(1).unwrap();
    }
    
    Matrix {
        data,
        rows: size,
        columns: size
    }
}



fn scale <T:Number>(m: &mut Matrix<T>, n: T) -> &mut Matrix<T> {
    
    m.data = m.data.iter().map(|x:&T| *x*n).collect();

    m

}



fn lu <T:Number>(A: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    //TODO edge cases - irregular matrices shapes
    //TODO rearrange rows to reduce numerical errors before computation begins, mangle indices, how and why ?
    //TODO handle pivot zero case - rows rearrangement - create indices map

    let size = A.rows * A.columns;
    
    let mut L: Matrix<T> = identity(A.rows);
    
    let mut U: Matrix<T> = A.clone();
    
    for i in 0..(U.rows - 1) {
        let p = U.data[((i * U.columns) + i) as usize]; 

        let mut tmp: Matrix<T> = identity(A.rows);

        for j in (i + 1)..U.rows {
            let e = U.data[((j * U.columns) + i) as usize];
            let c = T::from_i32(-1).unwrap() * (e/p);
            
            tmp.data[((j * U.columns) + i) as usize] = c; 
            
            for k in i..U.columns {
                let idx1 = ((i * U.columns) + k) as usize;
                let idx2 = ((j * U.columns) + k) as usize;
                U.data[idx2] = U.data[idx2] + (c * U.data[idx1]);
            }
        }
        
        let m = Matrix {
            data: tmp.data,
            rows: U.rows,
            columns: U.columns
        };
        
        //println!("L {} {:?}", i, m);

        L = multiply(&m, &L); //TODO improve
    }

    (L,U)
}



fn mul <T:Number>(
    A: &impl Index<[usize;2], Output = T>, 
    B: &impl Index<[usize;2], Output = T>,
    A_rows: usize,
    A_columns: usize,
    B_columns: usize
) -> Matrix<T> {
    
    let mut C: Matrix<T> = Matrix::new(A_rows, B_columns);
    
    for i in 0..A_rows {

        for j in 0..B_columns {

            for k in 0..A_columns {

                C[[i,j]] += A[[i,k]] * B[[k,j]];
            }
        }
    }

    C
}



pub fn multiply <T:Number>(A: &Matrix<T>, B: &Matrix<T>) -> Matrix<T> {

    assert_eq!(A.columns, B.rows, "matrices dimensions should be compatible A columns {} B rows {}", A.columns, B.rows);
    
    mul(A, B, A.rows, A.columns, B.columns)
}



pub fn strassen<T:Number>(A:Matrix<T>, B:Matrix<T>) -> Matrix<T> {
    
    let (
        A11, 
        A12, 
        A21, 
        A22
    ) = decompose_blocks(A);

    let (
        B11, 
        B12, 
        B21, 
        B22
    ) = decompose_blocks(B);
    
    let P1: Matrix<T> = &(&A11 + &A22) * &(&B11 + &B22); //TODO implement with consumption
    let P2: Matrix<T> = &(&A21 + &A22) * &B11;
    let P3: Matrix<T> = &A11 * &(&B12 - &B22);
    let P4: Matrix<T> = &A22 * &(&B21 - &B11);
    let P5: Matrix<T> = &(&A11 + &A12) * &B22;
    let P6: Matrix<T> = &(&A21 - &A11) * &(&B11 + &B12);
    let P7: Matrix<T> = &(&A12 - &A22) * &(&B21 + &B22);

    let C11: Matrix<T> = &(&(&P1 + &P4) - &P5) + &P7;
    let C12: Matrix<T> = &P3 + &P5;
    let C21: Matrix<T> = &P2 + &P4;
    let C22: Matrix<T> = &(&(&P1 + &P3) - &P2) + &P6;
    let C = recombine_blocks(C11, C12, C21, C22);
    
    C
}



pub fn eq_f64(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
    
    if a.rows != b.rows || a.columns != b.columns {
       return false;
    }

    for i in 0..a.rows {
        for j in 0..a.columns {
            if (a[[i,j]] - b[[i,j]]).abs() > EPSILON as f64 {
                return false;
            }
        } 
    }

    return true;
}



pub fn eq<T: Number>(a: &Matrix<T>, b: &Matrix<T>) -> bool {
    
    if a.rows != b.rows || a.columns != b.columns {
       return false;
    }

    for i in 0..a.rows {
        for j in 0..a.columns {
            if a[[i,j]] != b[[i,j]] {
                return false;
            }
        } 
    }

    return true;
}

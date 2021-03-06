#![allow(warnings)]
use crate::{Number, core::{matrix::{Matrix}, utils::{decompose_blocks, recombine_blocks, augment_sq2n, augment_sq2n_size}}};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;



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

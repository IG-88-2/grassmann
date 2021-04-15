#![allow(dead_code, warnings)]
use std::{cmp::min, collections::{HashMap, HashSet}, f32::EPSILON};
use super::matrix::{Matrix, add, P_compact, Partition};
use super::{vector::Vector};
use crate::{matrix, Number};
use super::{ 
    matrix3::Matrix3, 
    matrix4::Matrix4, 
    lu::{ 
        block_lu, 
        block_lu_threads_v2, 
        lu, 
        lu_v2
    },
    multiply::{ 
        multiply_threads,
        mul_blocks, 
        get_optimal_depth
    },
    utils::{
        eq_bound_eps_v, 
        eq_eps_f64, 
        eq_bound_eps, 
        eq_bound
    }
};



pub fn conjugate<T: Number>(A: &Matrix<T>, S: &Matrix<T>) -> Option<Matrix<T>> {

    let lu = S.lu();
    
    let S_inv = S.inv(&lu);
    
    if S_inv.is_none() {
       return None;
    }

    let inv = S_inv.unwrap();
    
    let K: Matrix<T> = &(&inv * A) * S;

    Some(K)
}
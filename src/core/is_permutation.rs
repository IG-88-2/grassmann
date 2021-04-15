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



pub fn is_permutation<T: Number>(A: &Matrix<T>) -> bool {

    if A.rows != A.columns || A.rows <= 1 {
       return false; 
    }

    for i in 0..(A.rows - 1) {

        let start = i + 1;

        for j in start..A.columns {
            
            if A[[i,j]] != A[[j,i]] {

                return false;
                
            }
        }
    }

    true
}

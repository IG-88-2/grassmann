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



pub fn from_basis<T: Number>(
    b: Vec<Vector<T>>
) -> Matrix<T> {
    
    if b.len() == 0 {
       return Matrix::new(0, 0);
    }

    let rows = b[0].data.len();

    let equal = b.iter().all(|v| v.data.len() == rows);
    
    assert!(equal, "\n from basis: vectors should have equal length \n");

    let columns = b.len();
    
    let mut m = Matrix::new(rows, columns);

    for i in 0..columns {
        let next = &b[i];
        for j in 0..rows {
            m[[j, i]] = next[j];
        }
    }

    m
}
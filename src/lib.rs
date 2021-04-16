#![allow(warnings, dead_code, unused_imports, unused)]
mod core;
mod functions;
mod workers;
use crate::core::*;
use num_traits::{Num, NumAssignOps, NumOps, PrimInt, Signed, cast::{FromPrimitive, ToPrimitive}, identities};
use std::fmt::{Debug}; 



pub type Float = f64;



pub trait Integer : PrimInt + Copy + NumAssignOps + Debug {}



pub trait Number : Num + Copy + ToString + FromPrimitive + ToPrimitive + NumOps + NumAssignOps + Debug + Sized + Signed + PartialOrd + 'static {
    fn to_ne_bytes(self) -> [u8; 8];
}



impl Integer for i32 {}



impl Number for i8 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..1]);
        a
    }
}



impl Number for i32 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..4]);
        a
    }
}



impl Number for f32 {
    fn to_ne_bytes(self) -> [u8; 8] {
        let mut a:[u8; 8] = [0; 8];
        let b = self.to_ne_bytes();
        a.copy_from_slice(&b[0..4]);
        a
    }
}



impl Number for f64 {
    fn to_ne_bytes(self) -> [u8; 8] {
        self.to_ne_bytes()
    }
}



pub struct Partition <T: Number> {
    pub A11: matrix::Matrix<T>,
    pub A12: matrix::Matrix<T>,
    pub A21: matrix::Matrix<T>,
    pub A22: matrix::Matrix<T>
}

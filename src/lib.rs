mod core;
mod utils;
mod tests;
use crate::core::*;
use num_traits::{Num, NumAssignOps, NumOps, PrimInt, Signed, cast, identities};



pub type Float = f64;



pub trait Integer : PrimInt + Copy + NumAssignOps + std::fmt::Debug {}



pub trait Number : Num + cast::FromPrimitive + Copy + NumOps + NumAssignOps + std::fmt::Debug + Sized + Signed + PartialOrd + 'static {
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

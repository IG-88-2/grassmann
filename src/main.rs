mod core;
mod tests;
mod workers;
use crate::core::*;
use num_traits::{Num, NumAssignOps, NumOps, PrimInt, Signed, cast::{FromPrimitive, ToPrimitive}, identities};
use std::{f64::consts::PI, fmt::{Debug}}; 
use crate::{vector::Vector, vector4::Vector4, vector3::Vector3, matrix3::Matrix3, matrix4::Matrix4, utils::{clamp, eq_eps_f64}};
use rand::prelude::*;
use rand::Rng;


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



fn main() {

    println!("hello...");
    let max = 50.;
    let mut rng = rand::thread_rng();
    let id = Matrix3::id();
    let b = id.into_basis();
    let id_x = b[0];
    let x = Vector3::rand(10.);
    let y = Vector3::rand(10.);
    let c1 = rng.gen_range(-max,max);
    let c2 = rng.gen_range(-max,max);
    let z = (&x * c1) + (&y * c2);

    let m = Matrix3::from_basis(x,y,z);

    assert!(eq_eps_f64(m.det(), 0.), "matrix should be singular");

    let orth: Matrix3 = Matrix3::orthonormal(&x);
    let rot: Matrix3 = Matrix4::rotation(PI / 4., 0., 0.).into();
    let t: Vector3 = &(orth * rot) * y;
    let tp: Vector3 = Matrix3::projection(&m, &t);
    let d = tp.distance(&t);
    
    assert!(d != 0., "t,tp distance should not be zero");

    println!("\n t is {} | tp is {} \n", t, tp);
    
    let e: Vector3 = t - tp;
    let d: f64 = e * y;

    println!("\n e is {} | y is {} | d is {} \n", e, y, d);

    assert!(eq_eps_f64(d, 0.), "d should be zero {}", d);

    //projection on full space is the same (try id)
    //e is orthogonal
    //project cross get zero
    //apply M
    //x angle with y id
    //x angle with z id
    
    //rotate y around x
    //rotate y in id space 
    //apply rotation which took id x to x 
    
    //id x vs x
    //relative to y,z
    //original location pi/2 with y,z
    //current location
    
    let dst: Vector3 = x - id_x;
    //which rotation is going to produce this effect ???
    let r = Matrix4::rotation(dst.x, dst.y, dst.z);

    //A through rotational composition
    //B through construction of orthonormal basis
    //x - id x - decompose into rotations
    //what rotation i need per dim

    //take x - vector from plane
    //rotate y around x
    //rotation
    // 

    //vector inside subspace - invariant
    //x angle with z axis
    //rotate y rotation by this angle
}
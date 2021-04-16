use crate::{Number, core::{matrix::Matrix, vector::Vector}};



pub fn project<T: Number>(A: &Matrix<T>, b:&Vector<T>) -> Vector<T> {

    assert_eq!(A.rows, b.data.len(), "A and b dimensions should correspond");

    let mut At = A.transpose(); //TODO transpose should behave, be called in the same way for all types (same applies for remaining methods)

    let Atb: Vector<T> = &At * b;

    let AtA: Matrix<T> = &At * A;
    
    let lu = AtA.lu();

    let x = AtA.solve(&Atb, &lu).unwrap();

    A * &x
}

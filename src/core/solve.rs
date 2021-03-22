use std::cmp::min;
use crate::Number;
use super::{lu::lu, matrix::Matrix, vector::Vector};



pub fn solve_lower_triangular<T:Number>(L:&Matrix<T>, b:&Vector<T>) -> Option<Vector<T>> {

    let zero = T::from_f64(0.).unwrap();

    let mut x = Vector::new(vec![zero; L.columns]);

    for i in 0..L.rows {
        let mut acc = b[i];

        for j in 0..(i + 1) {
            let c = L[[i, j]];
            
            if j == i {

                if c == zero {
                    let mut acc2 = zero;
    
                    for k in 0..j {
                        acc2 += x[k] * L[[i, k]];
                    }
    
                    let diff = (acc2 - b[i]).abs();
                    
                    if T::to_f32(&diff).unwrap().abs() < f32::EPSILON {
                        x[i] = zero;
                        continue;
                    } else {
                        return None;
                    }
                }
                
                x[i] = acc / c;

            } else {

                acc -= c * x[j];
            }
        }
    }

    Some(x)
}



pub fn solve_upper_triangular<T:Number>(U:&Matrix<T>, b:&Vector<T>) -> Option<Vector<T>> {

    let zero = T::from_f64(0.).unwrap();

    let mut x: Vector<T> = Vector::new(vec![zero; U.columns]);

    for i in (0..U.rows).rev() {
        let mut acc = b[i];
        
        let c = U[[i, i]];

        for j in (i..U.columns).rev() {
            if i == j {

                if c == zero {
                    let diff = acc.abs();
        
                    if T::to_f32(&diff).unwrap().abs() < f32::EPSILON {
                        x[i] = zero;
                        continue;
                    } else {
                        return None;
                    }
                }

                x[i] = acc / c;

            } else {

                acc -= U[[i, j]] * x[j];

            }
        }
    }

    Some(x)
}



pub fn solve<T:Number>(b: &Vector<T>, lu: &lu<T>) -> Option<Vector<T>> {
    let zero = T::from_f64(0.).unwrap();
    let one = T::from_f64(1.).unwrap();
    let Pb: Vector<T> = &lu.P * b;
    let PL: Matrix<T> = &lu.P * &lu.L;
    let mut y = Vector::new(vec![zero; PL.columns]);
    let mut x = Vector::new(vec![zero; lu.U.columns]);
    
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..lu.d.len() {
        let next = lu.d[i] as usize;
        x[next] = one;
    }

    for i in 0..x.data.len() {
        if x[i] == zero {
            indices.push(i as u32);
        }
    }
    
    for i in 0..PL.rows {
        let mut acc = Pb[i];

        for j in 0..(i + 1) {
            let c = PL[[i, j]];
            
            if j == i {

                if c == zero {
                    let mut acc2 = zero;
    
                    for k in 0..j {
                        acc2 += y[k] * PL[[i, k]];
                    }
    
                    let diff = (acc2 - Pb[i]).abs();
                    
                    if T::to_f32(&diff).unwrap().abs() < f32::EPSILON {
                        y[i] = zero;
                        continue;
                    } else {
                        return None;
                    }
                }
                
                y[i] = acc / c;

            } else {

                acc -= c * y[j];
            }
        }
    }
    
    let r = min(lu.U.rows, lu.U.columns) - lu.d.len();
    
    //println!("\n U is {} \n rhs is {} \n d is {:?} \n indices {:?} \n", lu.U, y, lu.d, indices);

    for i in (0..r).rev() {
        let mut acc = y[i];
        let target = indices[i] as usize; 
        let c = lu.U[[i, target]];

        for j in (i..lu.U.columns).rev() {
            if j == target {
                continue;
            }
            acc -= lu.U[[i, j]] * x[j];
        }

        if c == zero {
            let diff = (acc - y[i]).abs();

            if T::to_f32(&diff).unwrap().abs() < f32::EPSILON {
                x[target] = zero;
                continue;
            } else {
                return None;
            }
        }

        x[target] = acc / c;
    }
    
    Some(x)
}

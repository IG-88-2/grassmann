use std::cmp::min;
use crate::{Number, core::{matrix::Matrix, vector::Vector}};

use super::{lu::lu, utils::eq_eps_f64};



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



pub fn solve<T:Number>(b: &Vector<T>, lu: &lu<T>, tol: f64) -> Option<Vector<T>> {
    let zero = T::from_f64(0.).unwrap();
    let one = T::from_f64(1.).unwrap();
    let Pb: Vector<T> = &lu.P * b;
    let PL = &lu.P * &lu.L;
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
                    
                    if T::to_f64(&diff).unwrap() < tol {
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

            if T::to_f64(&diff).unwrap() < tol {
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



mod tests {

    use crate::{ Number, core::{matrix::{ Matrix }, vector::{ Vector }}, matrix, vector };
    use super::{ eq_eps_f64 };

    fn solve9() {
        let test = 50;

        for i in 1..test {
            println!("\n solve9: working with {} \n", i);

            let size = i;

            let max = 1000.;

            let A: Matrix<f64> = Matrix::rand_shape(size, max);
            
            let b: Vector<f64> = Vector::rand(A.rows as u32, max);
            
            let lu = A.lu();
            
            let x = A.solve(&b, &lu);
            
            if x.is_some() {

                let x = x.unwrap();

                println!("\n solved with x {} \n", x);

                let Ax = &A * &x;

                for j in 0..Ax.data.len() {
                    let eq = eq_eps_f64(Ax[j], b[j]);
                    if !eq {
                        println!("\n A is {} \n", A);
                        println!("\n x is {} \n", x);
                        println!("\n b is {} \n", b);
                        println!("\n Ax is {} \n", Ax);
                        println!("\n diff is {} \n", &Ax - &b);
                    }
                    assert!(eq, "entries should be equal");
                } 
            } else {
                println!("\n no solution! projecting\n");

                let b2 = A.project(&b);

                assert!(b2 != b, "b2 and b should be different");

                let x = A.solve(&b2, &lu);

                assert!(x.is_some(), "should be able to solve for projection");

                let x = x.unwrap();
                let Ax = &A * &x;

                for j in 0..Ax.data.len() {
                    let eq = eq_eps_f64(Ax[j], b2[j]);
                    if !eq {
                        println!("\n A is {} \n", A);
                        println!("\n x is {} \n", x);
                        println!("\n b2 is {} \n", b2);
                        println!("\n Ax is {} \n", Ax);
                        println!("\n diff is {} \n", &Ax - &b2);
                    }
                    assert!(eq, "entries should be equal");
                } 
            }
        }
    }



    fn solve8() {
        let test = 50;

        for i in 1..test {
            println!("\n solve: working with {} \n", i);

            let size = i;
            let max = 1000.;
            let A: Matrix<f64> = Matrix::rand(size, size, max);
            let b: Vector<f64> = Vector::rand(size as u32, max);
            let lu = A.lu();
            let x = A.solve(&b, &lu).unwrap();
            let Ax = &A * &x;

            println!("\n solve: Ax is {} \n b is {} \n", Ax, b);

            for j in 0..Ax.data.len() {
                let eq = eq_eps_f64(Ax[j], b[j]);
                if !eq {
                    println!("\n A is {} \n", A);
                    println!("\n x is {} \n", x);
                    println!("\n b is {} \n", b);
                    println!("\n Ax is {} \n", Ax);
                    println!("\n diff is {} \n", &Ax - &b);
                }
                assert!(eq, "entries should be equal");
            }  
        }
    }



    fn solve7() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2., 1., 2., 6.;
            1., 2., 1., 1., 1., 3.;
            1., 2., 3., 1., 3., 9.; 
        ];
        let b: Vector<f64> = vector![1., 0., 2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0],-3., "x0 is -3.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
        assert_eq!(x[3], 0., "x3 is  0.");
        assert_eq!(x[4], 0., "x4 is  0.");
        assert_eq!(x[5], 0., "x5 is  0.");
    }



    fn solve6() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 2., 1.;
            1., 2., 3.;
            1., 2., 3.;
            1., 2., 3.;
        ];

        let b: Vector<f64> = vector![2., 1., 3., 3., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);
        
        assert_eq!(x[0],-2., "x0 is -2.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
    }



    fn solve5() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 2., 1.;
            1., 2., 3.;
        ];
        let b: Vector<f64> = vector![2., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} , Ax is {} \n", x, &A * &x);
        
        assert_eq!(x[0],-2., "x0 is -2.");
        assert_eq!(x[1], 1., "x1 is  1.");
        assert_eq!(x[2], 1., "x2 is  1.");
    }



    fn solve4() {

        let A: Matrix<f64> = Matrix::id(3);

        let b: Vector<f64> = vector![3., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0], 3., "x0 is 3.");
        assert_eq!(x[1], 1., "x1 is 1.");
        assert_eq!(x[2], 3., "x2 is 3.");
    }



    fn solve3() {

        let A: Matrix<f64> = matrix![f64,
            1., 2., 2.;
            1., 1., 1.;
            1., 2., 3.;
        ];

        let b: Vector<f64> = vector![3., 1., 3.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        println!("\n x is {} \n", x);

        assert_eq!(x[0],-1., "x0 is -1");
        assert_eq!(x[1], 2., "x1 is  2");
        assert_eq!(x[2], 0., "x2 is  0");
    }



    fn solve2() {

        let A: Matrix<f64> = matrix![f64,
            1., 5., 7.;
        ];

        let b: Vector<f64> = vector![2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();
        
        assert_eq!(x[0], 2., "x0 is 2");
        assert_eq!(x[1], 0., "x1 is 0");
        assert_eq!(x[2], 0., "x2 is 0");
    }



    fn solve1() {

        let A: Matrix<f64> = matrix![f64,
            1.;
        ];

        let b: Vector<f64> = vector![2.];

        let lu = A.lu();

        let x = A.solve(&b, &lu).unwrap();

        assert_eq!(x[0], 2., "x is 2");
    }



    #[test]
    fn solve() {
       
       solve9();

       solve8();

       solve7();
       
       solve6();

       solve5();
    
       solve4();

       solve3();

       solve2();

       solve1();
    }
}
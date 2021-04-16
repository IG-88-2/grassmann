use crate::{Number, core::matrix::Matrix};



pub fn givens_theta<T: Number>(A: &Matrix<T>, i: usize, j: usize) -> f64 {

    let m = A[[i, j]];
    
    let d = A[[i - 1, j]];
    
    if d == T::from_f64(0.).unwrap() {
       return 0.;
    }

    let x: f64 = T::to_f64(&(m / d)).unwrap();
    
    let theta = x.atan();

    theta
}



pub fn givens<T: Number>(A: &Matrix<T>, i: usize, j: usize) -> Matrix<T> {
    
    let mut G: Matrix<T> = Matrix::id(A.rows);
    
    let theta = A.givens_theta(i, j);
    let s = T::from_f64( theta.sin() ).unwrap();
    let c = T::from_f64( theta.cos() ).unwrap();
    
    G[[i - 1, i - 1]] = c;
    G[[i, i - 1]] = -s;
    G[[i - 1, i]] = s;
    G[[i, i]] = c;
    
    G
}



pub fn givens3<T: Number>(A: &Matrix<T>, i: usize, j: usize) -> (Matrix<T>, Matrix<T>) {

    assert!(i > 0, "givens: i > 0");

    let mut G: Matrix<T> = Matrix::id(A.rows);
    let mut inv: Matrix<T> = Matrix::id(A.rows);
    
    let b = A[[i, j]];
    let e = A[[i + 1, j]];

    let x: f64 = T::to_f64(&(e / b)).unwrap();
    
    let phi = x.atan();
    let s = T::from_f64( phi.sin() ).unwrap();
    let c = T::from_f64( phi.cos() ).unwrap();
    
    println!("\n (f/e) is {}, (s/c) {}\n", x, (phi.sin() / phi.cos()));
    
    G[[i, j]] = c;
    G[[i + 1, j + 1]] = c;
    G[[i, j + 1]] = s;
    G[[i + 1, j]] = -s;
    
    let d = c * c + s * s;
    let k: T = c / d;
    let p: T = s / d;

    inv[[i, j]] = k;
    inv[[i + 1, j + 1]] = k;
    inv[[i, j + 1]] = -p;
    inv[[i + 1, j]] = p;
    
    (G, inv)
}



pub fn givens2<T: Number>(A: &Matrix<T>, i: usize, j: usize) -> Matrix<T> {

    assert!(i > 0, "givens: i > 0");

    let mut G: Matrix<T> = Matrix::id(A.rows);

    let basis = A.into_basis();

    //cos in which dimension

    let target = &basis[i];
    let mut target2 = target.clone();
    
    target2[j] = T::from_f64(0.).unwrap();

    let l1 = target.length();
    let l2 = target2.length();
    let d: f64 = (target * &target2) / (l1 * l2);
    let phi = d.acos();
    
    let s = T::from_f64( phi.sin() ).unwrap();
    let c = T::from_f64( phi.cos() ).unwrap();
    
    println!("\n givens2: target {} \n", target);
    println!("\n givens2: target2 {} \n", target2);
    println!("\n givens2: cos {}, c {:?}, s {:?} \n", d, c, s);
    
    G[[i, j]] = c;
    G[[i + 1, j + 1]] = c;
    G[[i, j + 1]] = s;
    G[[i + 1, j]] = -s;
    
    println!("\n givens2: G is {} \n", G);

    let Gt = G.transpose();

    println!("\n givens2: Gt is {} \n", &G * target);

    println!("\n givens2: Gtt is {} \n", &Gt * target);
    
    G
}

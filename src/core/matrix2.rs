//TODO move to matrix2
//#[test]
fn eig2x2_test() {

    let max = 6.3;
    let m = Matrix::rand(2, 2, max);
    let id = Matrix::id(2);
    let b = Vector::new(vec![0.,0.]);
    
    let e = Matrix::<f64>::eig2x2(&m).unwrap(); //?

    println!("\n A is {}, {} \n", m, m.rank());
    println!("\n e is {} {} \n", e.0, e.1);
    
    let A1 = &m - &(&id * e.0);
    let A2 = &m - &(&id * e.1);

    println!("\n A1 is {}, {} \n", A1, A1.rank());
    println!("\n A2 is {}, {} \n", A2, A2.rank());

    let lu_A1 = A1.lu();
    let x1 = A1.solve(&b, &lu_A1).unwrap();

    let lu_A2 = A2.lu();
    let x2 = A2.solve(&b, &lu_A2).unwrap();

    println!("\n x1 {} \n", x1);
    println!("\n x2 {} \n", x2);

    let y1: Vector<f64> = &A1 * &x1;
    let y2: Vector<f64> = &A1 * &x1;

    println!("\n y1 {} \n", y1);
    println!("\n y2 {} \n", y2);

    //rank
    //there is a vector in null space

    //assert!(false);
}

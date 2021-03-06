use crate::utils::{strassen::strassen};
use crate::core::matrix::{Matrix, Number, init_rand, augment_sq2n, augment_sq2n_size, eq_f64};
extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
use rand::prelude::*;
use rand::Rng;



#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}



pub fn get_test_matrices() -> (Matrix<f64>,Matrix<f64>) {
    let max = 50;
    let mut rng = rand::thread_rng();
    let A_rows = rng.gen_range(0, max) + 1; 
    let A_columns = rng.gen_range(0, max) + 1;
    let B_rows = A_columns;
    let B_columns = rng.gen_range(0, max) + 1;

    let mut A: Matrix<f64> = Matrix::new(A_rows, A_columns);
    let mut B: Matrix<f64> = Matrix::new(B_rows, B_columns);

    init_rand(&mut A);
    init_rand(&mut B);

    (A,B)
}



#[wasm_bindgen]
pub fn test_strassen() {

    console_error_panic_hook::set_once();

    let (mut a, mut b) = get_test_matrices();

    /*
    let a: Matrix<f64> = matrix![f64,
        3.,3.,1.,2.;
        3.,3.,1.,2.;
        4.,4.,5.,2.;
        4.,4.,5.,2.;
    ];

    let b: Matrix<f64> = matrix![f64,
        3.,3.,1.,2.;
        3.,3.,1.,2.;
        4.,4.,5.,2.;
        4.,4.,5.,2.;
    ];
    */
    
    let s1 = augment_sq2n_size(&a);

    let s2 = augment_sq2n_size(&b);

    let s = std::cmp::max(s1,s2);

    let mut A: Matrix<f64> = Matrix::new(s,s); 

    A = &A + &a;

    let mut B: Matrix<f64> = Matrix::new(s,s); 

    B = &B + &b;

    let expected: Matrix<f64> = &A * &B;

    unsafe {
        log(&format!("\n [{}] expected \n {:?} \n", expected.sum(), expected));
    }

    let C = strassen(A, B);

    unsafe {
        log(&format!("\n [{}] result \n {:?} \n", C.sum(), C));
    }

    let equal = eq_f64(&expected, &C);

    assert!(equal, "should be equal");
}
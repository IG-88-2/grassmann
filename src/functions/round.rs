use crate::{Number, core::matrix::Matrix};



pub fn round<T: Number>(A: &mut Matrix<T>, precision: f64) {

    let f = move |y: T| {
        let x = T::to_f64(&y).unwrap();
        let c = (2. as f64).powf(precision);
        T::from_f64((x * c).round() / c).unwrap()
    };

    A.apply(&f);
}

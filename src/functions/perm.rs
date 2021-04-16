use crate::core::matrix::Matrix;



//TODO improve, should be n!
pub fn perm(size: usize) -> Vec<Matrix<f32>> {
    let m: Matrix<f32> = Matrix::id(size);
    let mut list: Vec<Matrix<f32>> = Vec::new();
    
    for i in 0..size {
        for j in 0..size {
            if i != j {
                let mut k = Matrix::id(size);
                k.exchange_rows(i, j);
                list.push(k);
            } 
        }
    }

    let l = list.len();
    for i in 0..l {
        let A = &list[l - 1 - i];
        let P = &list[i];
        let p = A * P;
        list.push(p);
    }

    list
}
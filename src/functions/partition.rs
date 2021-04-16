use crate::{Number, Partition, core::matrix::Matrix};



pub fn partition<T: Number>(A: &Matrix<T>, r: usize) -> Option<Partition<T>> {
        
    if r >= A.columns || r >= A.rows {
        return None;
    }

    //A11 r x r
    //A12 r x (n - r)
    //A21 (n - r) x r
    //A22 (n - r) x (n - r)

    let mut A11: Matrix<T> = Matrix::new(r, r);
    let mut A12: Matrix<T> = Matrix::new(r, A.columns - r);
    let mut A21: Matrix<T> = Matrix::new(A.rows - r, r);
    let mut A22: Matrix<T> = Matrix::new(A.rows - r, A.columns - r);

    for i in 0..r {
        for j in 0..r {
            A11[[i,j]] = A[[i, j]];
        }
    }

    for i in 0..r {
        for j in 0..(A.columns - r) {
            A12[[i,j]] = A[[i,j + r]];
        }
    }

    for i in 0..(A.rows - r) {
        for j in 0..r {
            A21[[i,j]] = A[[i + r, j]];
        }
    }

    for i in 0..(A.rows - r) {
        for j in 0..(A.columns - r) {
            A22[[i,j]] = A[[i + r, j + r]];
        }
    }

    Some(
        Partition {
            A11,
            A12,
            A21,
            A22
        }
    )
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };


    #[test]
    fn partition() {

        let A = matrix![i32,
            1, 2;
            3, 4;
        ];

        let p = A.partition(1).unwrap();

        assert_eq!(p.A11.rows, 1, "p.A11.rows == 1");
        assert_eq!(p.A12.rows, 1, "p.A12.rows == 1");
        assert_eq!(p.A21.rows, 1, "p.A21.rows == 1");
        assert_eq!(p.A22.rows, 1, "p.A22.rows == 1");

        assert_eq!(p.A11.columns, 1, "p.A11.columns == 1");
        assert_eq!(p.A12.columns, 1, "p.A12.columns == 1");
        assert_eq!(p.A21.columns, 1, "p.A21.columns == 1");
        assert_eq!(p.A22.columns, 1, "p.A22.columns == 1");
        
        assert_eq!(p.A11[[0, 0]], 1, "p.A11[[0, 0]] == 1");
        assert_eq!(p.A12[[0, 0]], 2, "p.A12[[0, 0]] == 2");
        assert_eq!(p.A21[[0, 0]], 3, "p.A21[[0, 0]] == 3");
        assert_eq!(p.A22[[0, 0]], 4, "p.A22[[0, 0]] == 4");

        let A = matrix![i32,
            1, 2, 3, 6, 4, 6, 2, 5, 4;
            3, 4, 4, 4, 4, 5, 5, 5, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
        ];

        let p = A.partition(2).unwrap();

        println!("\n partition is \n A11 {} \n A12 {} \n A21 {} \n A22 {} \n", p.A11, p.A12, p.A21, p.A22);

        assert_eq!(p.A12.rows, 2, "p.A12.rows == 2");
        assert_eq!(p.A12.columns, 7, "p.A12.columns == 7");
    }
}

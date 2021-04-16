use crate::{Number, Partition, core::matrix::Matrix, matrix};



pub fn assemble<T: Number>(p: &Partition<T>) -> Matrix<T> {

    //A11 r x r
    //A12 r x (n - r)
    //A21 (n - r) x r
    //A22 (n - r) x (n - r)

    let rows = p.A11.rows + p.A21.rows;
    
    let columns = p.A11.columns + p.A12.columns; 
    
    let mut A = Matrix::new(rows, columns);
    
    for i in 0..p.A11.rows {
        for j in 0..p.A11.columns {
            A[[i, j]] = p.A11[[i, j]];
        }
    }

    for i in 0..p.A12.rows {
        for j in 0..p.A12.columns {
            A[[i, j + p.A11.columns]] = p.A12[[i, j]];
        }
    }
    
    for i in 0..p.A21.rows {
        for j in 0..p.A21.columns {
            A[[i + p.A11.rows, j]] = p.A21[[i, j]];
        }
    }

    for i in 0..p.A22.rows {
        for j in 0..p.A22.columns {
            A[[i + p.A11.rows, j + p.A11.columns]] = p.A22[[i, j]];
        }
    }

    A
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn assemble() {
        
        let A = matrix![i32,
            1, 2, 3, 6, 4, 6, 2, 5, 4;
            3, 4, 4, 4, 4, 5, 5, 5, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
            3, 4, 4, 3, 3, 5, 1, 1, 5;
        ];

        let p = A.partition(2).unwrap();

        let asm = Matrix::assemble(&p);

        assert_eq!(asm, A, "assemble 2: asm == A");

        let p = A.partition(3).unwrap();

        let asm = Matrix::assemble(&p);

        assert_eq!(asm, A, "assemble 3: asm == A");
    }
}

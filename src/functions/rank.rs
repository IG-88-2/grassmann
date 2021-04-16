use crate::{Number, core::matrix::Matrix};



pub fn rank<T: Number>(A: &Matrix<T>) -> u32 {

    if A.columns > A.rows {

        let At = A.transpose();

        let lu = At.lu();

        let rank = At.columns - lu.d.len();

        rank as u32

    } else {

        let lu = A.lu();

        let rank = A.columns - lu.d.len();

        rank as u32
    }
}



mod tests {

    use crate::{ core::{matrix::{ Matrix }}, matrix, vector };

    #[test]
    fn rank() {
        
        let mut A1 = matrix![f64,
            5.024026017784438, 2.858902178366669, 296.2138835869165;
            7.929129970221636, 5.7210492203315795, 523.7802005055211;
            8.85257084623291, 8.95057121546899, 704.1069012250204;
        ];

        let A2 = matrix![f64,
            1.1908477166794595, 8.793086722414468, 194.77132992778556;
            3.6478484951000456, 4.858421485429982, 187.58571816777294;
            9.423462238282756, 8.321761784861303, 406.23378670237366;
        ];

        let lu = A1.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        println!("\n d1 is {:?} \n A1 is {} \n R is {} \n L is {} \n U is {} \n diff is {} \n d is {:?} \n P is {} \n \n", lu.d, A1, R, lu.L, lu.U, &A1 - &R, lu.d, lu.P);

        assert_eq!(A1.rank(), 2, "A1 - rank is 2");
        
        let lu = A2.lu();

        let R: Matrix<f64> = &lu.L * &lu.U;

        println!("\n d2 is {:?} \n A2 is {} \n R is {} \n L is {} \n U is {} \n diff is {} \n d is {:?} \n P is {} \n \n", lu.d, A2, R, lu.L, lu.U, &A2 - &R, lu.d, lu.P);
        
        assert_eq!(A2.rank(), 2, "A2 - rank is 2");
        
        let test = 50;

        for i in 1..test {
            let max = 1.;

            let A: Matrix<f64> = Matrix::rand_shape(i, max);
            let At: Matrix<f64> = A.transpose();
            let AtA: Matrix<f64> = &At * &A;
            let AAt: Matrix<f64> = &A * &At;

            let rank_A = A.rank();
            let rank_AAt = AAt.rank();
            let rank_AtA = AtA.rank();

            let eq1 = rank_A == rank_AAt;
            let eq2 = rank_A == rank_AtA;

            if !eq1 {
                println!("\n rank A {}, rank AAt {} \n", rank_A, rank_AAt);
                println!("\n A ({}, {}) is {} \n AAt ({}, {}) {} \n", A.rows, A.columns, A, AAt.rows, AAt.columns, AAt);

                let luA = A.lu();
                let luAAt = AAt.lu();

                println!("\n U A is {} \n L A is {} \n d A is {:?} \n", luA.U, luA.L, luA.d);
                println!("\n U AAt is {} \n L AAt is {} \n d AAt is {:?} \n", luAAt.U, luAAt.L, luAAt.d);
            }

            if !eq2 {
                println!("\n rank A {}, rank AtA {} \n", rank_A, rank_AtA);
                println!("\n A ({}, {}) is {} \n AtA ({}, {}) {} \n", A.rows, A.columns, A, AtA.rows, AtA.columns, AtA);

                let luA = A.lu();
                let luAtA = AtA.lu();

                println!("\n U A is {} \n L A is {} \n d A is {:?} \n", luA.U, luA.L, luA.d);
                println!("\n U AtA is {} \n L AtA is {} \n d AtA is {:?} \n", luAtA.U, luAtA.L, luAtA.d);
            }
            //TODO rank AtA < rank A ??? fix
            assert!(eq1, "rank A and rank AAt should be equivalent");
            assert!(eq2, "rank A and rank AtA should be equivalent");
        }
    }
}
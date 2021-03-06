let rust = import('./out/grassmann');

rust.then((m) => {
    
    let hc = navigator.hardwareConcurrency;

    //m.test_multiplication(hc);
    //m.test_decomposition();
    //m.test_strassen();
    //m.test_matrix_mul_shader();
    
})
.catch((error) => {

    console.error(error);

});

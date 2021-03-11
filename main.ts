let rust = import('./out/grassmann');

rust.then((m) => {
    
    let hc = navigator.hardwareConcurrency;

    m.test_multiplication(hc);
    
})
.catch((error) => {

    console.error(error);

});

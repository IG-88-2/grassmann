let rust = import('./out/grassmann');

async function shim(m:any, hc:any) {
    const result = await m.test_multiplication(hc);
    return result
}

rust.then((m) => {
    
    let hc = navigator.hardwareConcurrency;
    
    return shim(m, hc); //m.test_multiplication(hc)
    
})
.then((result) => {
    console.log('result, ', result);
})
.catch((error) => {

    console.error(error);

});

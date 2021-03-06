let rs = import('./out/grassmann');

self.onmessage = function(args:any) {
    
    rs.then((m) => {

        //let sab = args.data[0];
        //let view = new Float64Array(sab);
        
        //let view2 = new Uint8Array(sab);

        //view[2] = 4.99;

        //let mem = m.wasm_memory();

        //i can transfer data here
        //call rust function and receive a) pointer to writable location b) size in bytes of writable location

        //const cells = new Uint8Array(mem.buffer);

        //console.log("module", rs, m, mem);

        console.log("data in worker", args.data);

        let sab = args.data[0];

        let a_rows = args.data[1];
        let a_columns = args.data[2];

        let b_rows = args.data[3];
        let b_columns = args.data[4];

        let t0 = args.data[5];
        let t1 = args.data[6];
        let t2 = args.data[7];
        let t3 = args.data[8];

        let t4 = args.data[9];
        let t5 = args.data[10];
        let t6 = args.data[11];
        let t7 = args.data[12];
        
        m.test_multiplication_worker(
            sab, 
            
            a_rows,
            a_columns,
            b_rows,
            b_columns,

            t0,
            t1,
            t2,
            t3,
    
            t4,
            t5,
            t6,
            t7
        );
        
        postMessage("done");

    });
}

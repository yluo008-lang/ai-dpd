// ============================================================================
// tb_nn_dpd_layer.v - checks nn_dpd_layer (layer 0) against the Python
// fixed-point reference (deploy/layer0_ref.txt produced by quant_export.py).
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
`include "req_params.vh"
module tb_nn_dpd_layer;
    localparam NIN = 12, NOUT = 32;
    reg clk=0, rst=1, ce=0;
    reg  [NIN*16-1:0]  xin = 0;
    wire [NOUT*16-1:0] yout;
    integer req0, i, k, err=0, fd, rc;
    integer xq [0:NIN-1];
    integer eq [0:NOUT-1];

    nn_dpd_layer #(.NIN(NIN),.NOUT(NOUT),.REQ(`REQ0),.SHIFT(`SHIFTQ),.RELU(1),.WFILE("w0.mem"),.BFILE("b0.mem")) dut (
        .clk(clk), .rst(rst), .ce(ce), .xin(xin), .yout(yout));

    always #5 clk = ~clk;

    initial begin
        fd = $fopen("req.txt","r"); rc = $fscanf(fd,"%d %d\n", req0, i); $fclose(fd);
        rst=1; ce=0; repeat(4) @(negedge clk); rst=0; ce=1;
        fd = $fopen("layer0_ref.txt","r");
        for (i=0; i<16; i=i+1) begin
            @(negedge clk);
            rc = $fscanf(fd,"%d %d %d %d %d %d %d %d %d %d %d %d",
                xq[0],xq[1],xq[2],xq[3],xq[4],xq[5],xq[6],xq[7],xq[8],xq[9],xq[10],xq[11]);
            for (k=0;k<NIN;k=k+1) xin[k*16 +: 16] = xq[k][15:0];
            for (k=0;k<NOUT;k=k+1) rc = $fscanf(fd,"%d",eq[k]);
            @(negedge clk);
            for (k=0;k<NOUT;k=k+1)
                if ($signed(yout[k*16 +: 16]) !== eq[k]) err = err+1;
        end
        $fclose(fd);
        $display("nn_dpd_layer: REQ0(header)=%0d  mismatches=%0d", req0, err);
        if (err==0) $display("PASS: layer0 matches Python fixed-point reference");
        else        $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

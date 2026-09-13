// ============================================================================
// nn_dpd_layer.v - streaming dense layer (int16 weights/activations) + ReLU
//
//   acc = sum_k W[o][k]*x[k]                    (int64 accumulate)
//   y   = saturate( (acc * REQ) >>> SHIFT )     (REQ = (sw*sx/sy)*2^SHIFT)
//   optional ReLU
//
// Weights are preloaded from a $readmemh image (w0.mem/w1.mem/w2.mem exported
// by quant_export.py). Fully parallel MACs, 1 output vector / clock.
// ============================================================================
`default_nettype none
module nn_dpd_layer #(
    parameter NIN  = 12,
    parameter NOUT = 32,
    parameter REQ  = 16384,
    parameter SHIFT = 30,
    parameter RELU = 1,
    parameter WFILE = "w0.mem",
    parameter BFILE = "b0.mem"
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                ce,
    input  wire [NIN*16-1:0]   xin,      // packed int16 inputs
    output reg  [NOUT*16-1:0]  yout
);
    reg signed [15:0] W  [0:NOUT*NIN-1];
    reg signed [31:0] BQ [0:NOUT-1];
    initial begin
        $readmemh(WFILE, W);
        $readmemh(BFILE, BQ);
    end

    wire signed [15:0] xv [0:NIN-1];
    genvar g;
    generate for (g=0; g<NIN; g=g+1) assign xv[g] = $signed(xin[g*16 +: 16]); endgenerate

    reg signed [63:0] acc [0:NOUT-1];
    reg signed [63:0] rq  [0:NOUT-1];
    reg signed [15:0] yo  [0:NOUT-1];
    integer o, k;

    always @(*) begin
        for (o=0; o<NOUT; o=o+1) begin
            acc[o] = 0;
            for (k=0; k<NIN; k=k+1)
                acc[o] = acc[o] + (W[o*NIN+k] * xv[k]);
            rq[o] = (acc[o]*REQ + ($signed(BQ[o]) <<< SHIFT) + (1 <<< (SHIFT-1))) >>> SHIFT;
            if (RELU && rq[o] < 0) rq[o] = 0;
            if (rq[o] >  32767) rq[o] =  32767;
            if (rq[o] < -32768) rq[o] = -32768;
            yo[o] = rq[o][15:0];
        end
    end

    always @(posedge clk) begin
        if (rst) yout <= 0;
        else if (ce) begin
            for (o=0; o<NOUT; o=o+1) yout[o*16 +: 16] <= yo[o];
        end
    end
endmodule
`default_nettype wire

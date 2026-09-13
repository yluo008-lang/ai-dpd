// ============================================================================
// nn_dpd_top.v - 3-layer MLP DPD datapath:  features -> L0 -> L1 -> L2(+residual)
//
//   xin : NIN int16 features per sample  [I_0,Q_0,|x|_0, I_1,Q_1,|x|_1, ...]
//   yout: 2 int16 outputs of the correction network (scale = SX2/SW2)
//
// The real DPD output is  z = yout*SX_last + x(n)  (residual added downstream,
// where x(n) = (features[0], features[1]) is available). Latency 3 cycles.
// Weights are $readmemh images from quant_export.py.
// ============================================================================
`default_nettype none
module nn_dpd_top #(
    parameter NIN = 12,
    parameter H   = 32,
    parameter REQ0 = 16384,
    parameter REQ1 = 16384,
    parameter SHIFT = 30
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                ce,
    input  wire [NIN*16-1:0]   xin,
    output wire [2*16-1:0]     ycorr            // NN correction (int16 x2)
);
    wire [H*16-1:0]  a1, a2;

    nn_dpd_layer #(.NIN(NIN),.NOUT(H),.REQ(REQ0),.SHIFT(SHIFT),.RELU(1),.WFILE("w0.mem"),.BFILE("b0.mem")) L0 (
        .clk(clk), .rst(rst), .ce(ce), .xin(xin), .yout(a1));

    nn_dpd_layer #(.NIN(H),.NOUT(H),.REQ(REQ1),.SHIFT(SHIFT),.RELU(1),.WFILE("w1.mem"),.BFILE("b1.mem")) L1 (
        .clk(clk), .rst(rst), .ce(ce), .xin(a1), .yout(a2));

    // output layer: linear, no requant -> raw accumulator (scale = SW2*SX2)
    nn_dpd_layer #(.NIN(H),.NOUT(2),.REQ(1),.SHIFT(0),.RELU(0),.WFILE("w2.mem"),.BFILE("b2.mem")) L2 (
        .clk(clk), .rst(rst), .ce(ce), .xin(a2), .yout(ycorr));
endmodule
`default_nettype wire

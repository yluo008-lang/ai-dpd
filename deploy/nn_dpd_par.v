// ============================================================================
// nn_dpd_par.v - P-sample/clock (polyphase) neural DPD datapath.
//
//   xin  : P consecutive complex samples (P*32 bit, {Q,I}, sample0 in LSB)
//   yout : P corrections, one per lane (P*32 bit)
//
// Per lane: nn_dpd_feat_par builds the 12 features from a shared history shift
// register, then 3 pipelined nn_dpd_layer (w0/w1/w2). Replicating the (already
// verified) MLP P times is the trivial part; the shared delay line is the hard
// part and lives in nn_dpd_feat_par.
// ============================================================================
`default_nettype none
module nn_dpd_par #(
    parameter P  = 2,
    parameter M  = 4,
    parameter IW = 16,
    parameter H  = 32,
    parameter REQ0 = 16384,
    parameter REQ1 = 16384,
    parameter SHIFT = 30
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              ce,
    input  wire [P*2*IW-1:0] xin,
    output wire [P*2*IW-1:0] yout
);
    wire [P*3*M*IW-1:0] feats;
    nn_dpd_feat_par #(.P(P), .M(M), .IW(IW)) u_feat (
        .clk(clk), .rst(rst), .ce(ce), .xin(xin), .feats(feats));

    genvar g;
    generate
        for (g=0; g<P; g=g+1) begin : lane
            wire [H*IW-1:0]   a1, a2;
            wire [2*IW-1:0]   corr;
            nn_dpd_layer #(.NIN(3*M),.NOUT(H),.REQ(REQ0),.SHIFT(SHIFT),.RELU(1),
                           .WFILE("w0.mem"),.BFILE("b0.mem")) L0 (
                .clk(clk), .rst(rst), .ce(ce),
                .xin(feats[g*3*M*IW +: 3*M*IW]), .yout(a1));
            nn_dpd_layer #(.NIN(H),.NOUT(H),.REQ(REQ1),.SHIFT(SHIFT),.RELU(1),
                           .WFILE("w1.mem"),.BFILE("b1.mem")) L1 (
                .clk(clk), .rst(rst), .ce(ce), .xin(a1), .yout(a2));
            nn_dpd_layer #(.NIN(H),.NOUT(2),.REQ(1),.SHIFT(0),.RELU(0),
                           .WFILE("w2.mem"),.BFILE("b2.mem")) L2 (
                .clk(clk), .rst(rst), .ce(ce), .xin(a2), .yout(corr));
            assign yout[g*2*IW +: 2*IW] = corr;
        end
    endgenerate
endmodule
`default_nettype wire

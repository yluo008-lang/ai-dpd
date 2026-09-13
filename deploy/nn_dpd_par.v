// ============================================================================
// nn_dpd_par.v - P-sample/clock (polyphase) neural DPD datapath, FULL chain.
//
//   xin  : P complex samples {Q,I}, sample0 in LSB (P*32 bit)
//   yout : P DPD outputs {Q,I} (P*32 bit)  = MLP correction (Q1.15) + delayed x
//
// Lane: feat_par(shared history) -> scale -> L0 -> L1 -> L2 -> +x(delayed 6)
// (L0/L1/L2 are each 2-cycle pipelined => input delayed 6 for the residual add)
// ============================================================================
`default_nettype none
module nn_dpd_par #(
    parameter P  = 2,
    parameter M  = 4,
    parameter IW = 16,
    parameter H  = 32,
    parameter REQ0 = 16384,
    parameter REQ1 = 16384,
    parameter REQ2 = 2048,
    parameter SHIFT = 30
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              ce,
    input  wire [P*2*IW-1:0] xin,
    output reg  [P*2*IW-1:0] yout
);
    wire [P*3*M*IW-1:0] feats, fsc;
    nn_dpd_feat_par #(.P(P), .M(M), .IW(IW)) u_feat (
        .clk(clk), .rst(rst), .ce(ce), .xin(xin), .feats(feats));
    nn_dpd_scale #(.NF(P*3*M)) u_scale (.fin(feats), .fout(fsc));

    // input delay line (6 cycles) for the residual add
    reg [P*2*IW-1:0] xd [0:5];
    integer i;
    always @(posedge clk) begin
        if (rst) for (i=0;i<6;i=i+1) xd[i] <= 0;
        else if (ce) begin xd[0] <= xin; for (i=1;i<6;i=i+1) xd[i] <= xd[i-1]; end
    end

    function signed [IW-1:0] sat;
        input signed [IW:0] v;
        begin
            if (v >  32767) sat =  32767;
            else if (v < -32768) sat = -32768;
            else sat = v[IW-1:0];
        end
    endfunction

    genvar g;
    generate
        for (g=0; g<P; g=g+1) begin : lane
            wire [H*IW-1:0] a1, a2;
            wire [2*IW-1:0] corr;
            nn_dpd_layer #(.NIN(3*M),.NOUT(H),.REQ(REQ0),.SHIFT(SHIFT),.RELU(1),
                           .WFILE("w0.mem"),.BFILE("b0.mem")) L0 (
                .clk(clk), .rst(rst), .ce(ce),
                .xin(fsc[g*3*M*IW +: 3*M*IW]), .yout(a1));
            nn_dpd_layer #(.NIN(H),.NOUT(H),.REQ(REQ1),.SHIFT(SHIFT),.RELU(1),
                           .WFILE("w1.mem"),.BFILE("b1.mem")) L1 (
                .clk(clk), .rst(rst), .ce(ce), .xin(a1), .yout(a2));
            nn_dpd_layer #(.NIN(H),.NOUT(2),.REQ(REQ2),.SHIFT(SHIFT),.RELU(0),
                           .WFILE("w2.mem"),.BFILE("b2.mem")) L2 (
                .clk(clk), .rst(rst), .ce(ce), .xin(a2), .yout(corr));
            wire signed [IW-1:0] ci = corr[0*IW +: IW];
            wire signed [IW-1:0] cq = corr[1*IW +: IW];
            wire signed [IW-1:0] xi = xd[5][(2*g)*IW   +: IW];
            wire signed [IW-1:0] xq = xd[5][(2*g+1)*IW +: IW];
            always @(*) begin
                yout[(2*g)*IW   +: IW] = sat({ci[IW-1], ci} + {xi[IW-1], xi});
                yout[(2*g+1)*IW +: IW] = sat({cq[IW-1], cq} + {xq[IW-1], xq});
            end
        end
    endgenerate
endmodule
`default_nettype wire

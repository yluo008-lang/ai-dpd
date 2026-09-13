// ============================================================================
// nn_dpd_feat.v - feature builder: complex stream -> [I_m, Q_m, |x_m|] per tap
//
//   tap m value = x(n-m);  |x| = integer sqrt(I^2+Q^2)  (Q1.15, matches Python)
//   Features are emitted for the current input sample; a causal delay line of
//   depth M holds the history. Integer sqrt uses a 16-step binary search.
// ============================================================================
`default_nettype none
module nn_dpd_feat #(
    parameter M  = 4,
    parameter IW = 16
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                ce,
    input  wire signed [IW-1:0] xi,
    input  wire signed [IW-1:0] xq,
    output wire [3*M*IW-1:0]   feats        // { ... |x|_0, Q_0, I_0 }  (tap0 = LSB)
);
    reg signed [IW-1:0] di [0:M-1], dq [0:M-1];
    integer m;
    always @(posedge clk) begin
        if (rst) for (m=0;m<M;m=m+1) begin di[m]<=0; dq[m]<=0; end
        else if (ce) begin
            di[0] <= xi; dq[0] <= xq;
            for (m=1;m<M;m=m+1) begin di[m] <= di[m-1]; dq[m] <= dq[m-1]; end
        end
    end

    // integer sqrt (Q1.15): |x| = floor(sqrt(I^2+Q^2))
    function [IW-1:0] isqrt32;
        input [2*IW-1:0] v;                  // I^2+Q^2 fits in 32 bits
        integer i;
        reg [2*IW-1:0] r;
        reg [2*2*IW-1:0] sq;                 // (r|bit)^2
        reg [2*IW-1:0] t;
        begin
            r = 0;
            for (i=IW-1; i>=0; i=i-1) begin
                t  = r | (32'd1 << i);
                sq = t*t;
                if (sq <= v) r = t;
            end
            isqrt32 = r[IW-1:0];
        end
    endfunction

    genvar g;
    generate
        for (g=0; g<M; g=g+1) begin : gtap
            // constant-safe index: g=0 -> index 0 (value unused), else g-1
            wire signed [IW-1:0] ti = (g == 0) ? xi : di[(g == 0) ? 0 : (g-1)];
            wire signed [IW-1:0] tq = (g == 0) ? xq : dq[(g == 0) ? 0 : (g-1)];
            wire [2*IW-1:0] mag2 = ti*ti + tq*tq;
            wire [IW-1:0]  mag   = isqrt32(mag2);
            assign feats[(3*g+0)*IW +: IW] = ti;
            assign feats[(3*g+1)*IW +: IW] = tq;
            assign feats[(3*g+2)*IW +: IW] = mag;
        end
    endgenerate
endmodule
`default_nettype wire

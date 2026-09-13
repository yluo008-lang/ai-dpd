// ============================================================================
// nn_dpd_feat_par.v - PARALLEL feature builder: P samples/clock.
//
// One shared history register feeds P lanes; lane p emits the 12 features of the
// sample x(n-p) .. using taps x(n-p-k). Replicating the MLP P times on these
// lanes gives a P-sample/clock (polyphase) DPD front-end at 1/P the clock rate.
//
// combined window comb[0..M-1+P-1] = [history (M-1)][new P]; lane p sample is
// comb[M-1+p], tap k is comb[M-1+p-k].
// ============================================================================
`default_nettype none
module nn_dpd_feat_par #(
    parameter P  = 2,
    parameter M  = 4,
    parameter IW = 16
)(
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  ce,
    input  wire [P*2*IW-1:0]     xin,       // P samples {Q,I}; sample0 in LSB
    output wire [P*3*M*IW-1:0]   feats      // lane0 features in LSB
);
    localparam CW = (M-1) + P;              // combined window length

    reg signed [IW-1:0] hi [0:M-2], hq [0:M-2];   // history (M-1 entries)
    wire signed [IW-1:0] ni [0:P-1], nq [0:P-1];
    wire signed [IW-1:0] ci [0:CW-1], cq [0:CW-1];

    genvar g, k;
    generate
        for (g=0; g<P; g=g+1) begin : gnew
            assign ni[g] = $signed(xin[(2*g)*IW   +: IW]);
            assign nq[g] = $signed(xin[(2*g+1)*IW +: IW]);
        end
        for (g=0; g<CW; g=g+1) begin : gcomb
            if (g < M-1) begin
                assign ci[g] = hi[g];  assign cq[g] = hq[g];
            end else begin
                assign ci[g] = ni[g-(M-1)];  assign cq[g] = nq[g-(M-1)];
            end
        end
    endgenerate

    function [IW-1:0] isqrt32;
        input [2*IW-1:0] v;
        integer i; reg [2*IW-1:0] r; reg [2*2*IW-1:0] sq; reg [2*IW-1:0] t;
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

    // per-lane features (tap k = comb[M-1+p-k])
    generate
        for (g=0; g<P; g=g+1) begin : glane
            for (k=0; k<M; k=k+1) begin : gtap
                wire signed [IW-1:0] ti = ci[(M-1)+g-k];
                wire signed [IW-1:0] tq = cq[(M-1)+g-k];
                wire [2*IW-1:0] mag2 = ti*ti + tq*tq;
                wire [IW-1:0]  mag   = isqrt32(mag2);
                assign feats[(g*3*M + 3*k + 0)*IW +: IW] = ti;
                assign feats[(g*3*M + 3*k + 1)*IW +: IW] = tq;
                assign feats[(g*3*M + 3*k + 2)*IW +: IW] = mag;
            end
        end
    endgenerate

    integer c;
    always @(posedge clk) begin
        if (rst) for (c=0;c<M-1;c=c+1) begin hi[c]<=0; hq[c]<=0; end
        else if (ce) for (c=0;c<M-1;c=c+1) begin
            hi[c] <= ci[P+c];  hq[c] <= cq[P+c];      // newest M-1 samples
        end
    end
endmodule
`default_nettype wire

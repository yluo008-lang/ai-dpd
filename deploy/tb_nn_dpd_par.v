// ============================================================================
// tb_nn_dpd_par.v - parallel chain: iq -> feat_par -> scale -> L0  vs Python golden.
//
// NOTE: xin MUST advance every clock (ce is always 1), otherwise the shared
// history register in nn_dpd_feat_par shifts more than once per group. So we
// stream one group per cycle and capture outputs with the L0 pipeline latency.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
`include "req_params.vh"
`include "par_params.vh"
module tb_nn_dpd_par;
    localparam P = 2, M = 4, IW = 16, H = 32, NF = 3*M, NV = 64, NG = NV/P;
    reg clk=0, rst=1, ce=0;
    reg  [P*2*IW-1:0] xin = 0;
    wire [P*NF*IW-1:0] feats, fsc;
    wire [P*H*IW-1:0]  a1;
    integer t, p, k, s, fd_iq, fd_rf, rc, err=0;
    reg  [15:0] iv [0:NG*P-1], qv [0:NG*P-1];
    reg signed [15:0] exp [0:NG*P*H-1];
    reg signed [15:0] got [0:NG*P*H-1];

    nn_dpd_feat_par #(.P(P),.M(M),.IW(IW)) u_feat (.clk(clk),.rst(rst),.ce(ce),.xin(xin),.feats(feats));
    nn_dpd_scale #(.NF(P*NF)) u_scale (.fin(feats), .fout(fsc));
    genvar g;
    generate
        for (g=0; g<P; g=g+1) begin : lane
            nn_dpd_layer #(.NIN(NF),.NOUT(H),.REQ(`REQ0),.SHIFT(`SHIFTQ),.RELU(1),
                           .WFILE("w0.mem"),.BFILE("b0.mem")) L0 (
                .clk(clk),.rst(rst),.ce(ce),
                .xin(fsc[g*NF*IW +: NF*IW]), .yout(a1[g*H*IW +: H*IW]));
        end
    endgenerate

    always #5 clk = ~clk;

    initial begin
        fd_iq = $fopen("iq.hex","r"); fd_rf = $fopen("par_ref.hex","r");
        if (fd_iq==0 || fd_rf==0) begin $display("ERROR: vectors missing"); $finish; end
        for (s=0; s<NG*P;   s=s+1) rc = $fscanf(fd_iq,"%h %h\n", iv[s], qv[s]);
        for (s=0; s<NG*P*H; s=s+1) rc = $fscanf(fd_rf,"%h", exp[s]);
        $fclose(fd_iq); $fclose(fd_rf);

        rst=1; ce=0; repeat(4) @(negedge clk); rst=0; ce=1;
        for (t=0; t<NG+2; t=t+1) begin
            @(negedge clk);
            if (t<NG) begin
                xin = 0;
                for (p=0;p<P;p=p+1) begin
                    xin[(2*p)*IW +: IW]   = iv[t*P+p];
                    xin[(2*p+1)*IW +: IW] = qv[t*P+p];
                end
            end
            if (t>=2)                                  // output ~2 cycles after its group
                for (p=0;p<P;p=p+1) for (k=0;k<H;k=k+1)
                    got[((t-2)*P+p)*H+k] = $signed(a1[(p*H+k)*IW +: IW]);
        end
        for (s=0; s<NG*P*H; s=s+1)
            if (got[s] !== exp[s]) err = err+1;
        $display("nn_dpd_par chain (feat->scale->L0, P=%0d): mismatches = %0d / %0d", P, err, NG*P*H);
        if (err==0) $display("PASS: parallel DPD chain matches Python golden");
        else        $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

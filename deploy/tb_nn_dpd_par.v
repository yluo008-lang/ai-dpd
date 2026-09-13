// ============================================================================
// tb_nn_dpd_par.v - FULL parallel chain: iq -> feat->scale->L0->L1->L2 + residual
// vs Python golden (par_ref.hex: {I_out,Q_out} per sample). P samples/clock.
// xin advances every clock (ce=1) so the shared history shifts once per group.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
`include "req_params.vh"
`include "par_params.vh"
module tb_nn_dpd_par;
    localparam P = 2, M = 4, IW = 16, H = 32, NF = 3*M, NV = 64, NG = NV/P;
    reg clk=0, rst=1, ce=0;
    reg  [P*2*IW-1:0] xin = 0;
    wire [P*2*IW-1:0] yout;
    integer t, p, s, L, fd_iq, fd_rf, rc, err=0, e2, best=1000000, bestL=-1;
    reg [15:0] capr [0:(NG+8)*P-1], capi [0:(NG+8)*P-1];
    reg  [15:0] iv [0:NG*P-1], qv [0:NG*P-1];
    reg  [15:0] er [0:NG*P-1], ei [0:NG*P-1];
    reg  [15:0] gr [0:NG*P-1], gi [0:NG*P-1];

    nn_dpd_par #(.P(P),.M(M),.IW(IW),.H(H),.REQ0(`REQ0),.REQ1(`REQ1),.REQ2(`REQ2),.SHIFT(`SHIFTQ)) dut (
        .clk(clk),.rst(rst),.ce(ce),.xin(xin),.yout(yout));

    always #5 clk = ~clk;

    initial begin
        fd_iq = $fopen("iq.hex","r"); fd_rf = $fopen("par_ref.hex","r");
        if (fd_iq==0 || fd_rf==0) begin $display("ERROR: vectors missing"); $finish; end
        for (s=0; s<NG*P; s=s+1) rc = $fscanf(fd_iq,"%h %h\n", iv[s], qv[s]);
        for (s=0; s<NG*P; s=s+1) rc = $fscanf(fd_rf,"%h %h\n", er[s], ei[s]);
        $fclose(fd_iq); $fclose(fd_rf);

        rst=1; ce=0; repeat(4) @(negedge clk); rst=0; ce=1;
        for (t=0; t<NG+8; t=t+1) begin
            @(negedge clk);
            if (t<NG) begin
                xin = 0;
                for (p=0;p<P;p=p+1) begin
                    xin[(2*p)*IW +: IW]   = iv[t*P+p];
                    xin[(2*p+1)*IW +: IW] = qv[t*P+p];
                end
            end
            for (p=0;p<P;p=p+1) begin
                capr[t*P+p] = yout[(2*p)*IW   +: IW];
                capi[t*P+p] = yout[(2*p+1)*IW +: IW];
            end
        end
        for (L=4; L<=9; L=L+1) begin
            e2 = 0;
            for (t=L; t<NG; t=t+1) for (p=0;p<P;p=p+1) begin
                if (capr[t*P+p] !== er[(t-L)*P+p]) e2 = e2+1;
                if (capi[t*P+p] !== ei[(t-L)*P+p]) e2 = e2+1;
            end
            $display("  latency %0d: mismatches = %0d", L, e2);
            if (bestL<0 || e2<best) begin best=e2; bestL=L; end
        end
        err = best;
        $display("nn_dpd_par FULL chain (P=%0d): best latency %0d, mismatches = %0d", P, bestL, err);
        if (err==0) $display("PASS: full parallel DPD chain matches Python golden");
        else        $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

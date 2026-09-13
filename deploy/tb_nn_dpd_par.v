// ============================================================================
// tb_nn_dpd_par.v - verifies the PARALLEL chain: iq -> feat_par -> scale -> L0
// against the Python golden (par_ref.hex), lane p of group t == line t*P+p.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
`include "req_params.vh"
`include "par_params.vh"
module tb_nn_dpd_par;
    localparam P = 2, M = 4, IW = 16, H = 32, NF = 3*M, NV = 64;
    reg clk=0, rst=1, ce=0;
    reg  [P*2*IW-1:0] xin = 0;
    wire [P*NF*IW-1:0] feats;
    wire [P*NF*IW-1:0] fsc;
    wire [P*H*IW-1:0]  a1;
    integer t, p, k, fd_iq, fd_rf, rc, err=0;
    reg [15:0] a [0:P-1], b [0:P-1], e [0:H-1];

    nn_dpd_feat_par #(.P(P),.M(M),.IW(IW)) u_feat (.clk(clk),.rst(rst),.ce(ce),.xin(xin),.feats(feats));
    nn_dpd_scale #(.NF(P*NF)) u_scale (.fin(feats), .fout(fsc));
    generate genvar g;
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
        rst=1; ce=0; repeat(4) @(negedge clk); rst=0; ce=1;
        @(negedge clk);
        for (t=0; t<NV/P; t=t+1) begin
            @(negedge clk);
            for (p=0;p<P;p=p+1) begin
                rc = $fscanf(fd_iq,"%h %h\n",a[p],b[p]);
                xin[(2*p)*IW +: IW]=a[p]; xin[(2*p+1)*IW +: IW]=b[p];
            end
            // feat (comb) + scale (comb) + L0 (2-cycle pipeline)
            @(negedge clk); @(negedge clk);
            for (p=0;p<P;p=p+1) begin
                for (k=0;k<H;k=k+1) rc = $fscanf(fd_rf,"%h",e[k]);   // this lane's golden
                for (k=0;k<H;k=k+1)
                    if ($signed(a1[(p*H+k)*IW +: IW]) !== $signed(e[k])) err = err+1;
            end
        end
        $fclose(fd_iq); $fclose(fd_rf);
        $display("nn_dpd_par chain (feat->scale->L0, P=%0d): mismatches = %0d", P, err);
        if (err==0) $display("PASS: parallel DPD chain matches Python golden");
        else        $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

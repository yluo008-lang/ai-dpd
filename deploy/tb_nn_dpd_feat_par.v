// ============================================================================
// tb_nn_dpd_feat_par.v - checks the P-wide parallel feature builder against the
// per-sample Python golden features (feat.hex), lane p of group t == line t*P+p.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
module tb_nn_dpd_feat_par;
    localparam P = 2, M = 4, IW = 16, NF = 3*M, NV = 64;
    reg clk = 0, rst = 1, ce = 0;
    reg  [P*2*IW-1:0] xin = 0;
    wire [P*NF*IW-1:0] feats;
    integer t, p, k, fd_iq, fd_ft, rc, err = 0;
    reg [15:0] a [0:P-1], b [0:P-1], e [0:NF-1];

    nn_dpd_feat_par #(.P(P), .M(M), .IW(IW)) dut (
        .clk(clk), .rst(rst), .ce(ce), .xin(xin), .feats(feats));

    always #5 clk = ~clk;

    initial begin
        fd_iq = $fopen("iq.hex", "r");
        fd_ft = $fopen("feat.hex", "r");
        if (fd_iq == 0 || fd_ft == 0) begin $display("ERROR: vectors missing"); $finish; end
        rst = 1; ce = 0; repeat (3) @(negedge clk); rst = 0; ce = 1;
        @(negedge clk);
        for (t = 0; t < NV/P; t = t+1) begin
            // pack P input samples for group t
            for (p = 0; p < P; p = p+1) begin
                rc = $fscanf(fd_iq, "%h %h\n", a[p], b[p]);
                xin[(2*p)*IW +: IW]   = a[p];
                xin[(2*p+1)*IW +: IW] = b[p];
            end
            #1;
            // each lane p must equal feat.hex line (t*P+p)
            for (p = 0; p < P; p = p+1) begin
                for (k = 0; k < NF; k = k+1) rc = $fscanf(fd_ft, "%h", e[k]);
                for (k = 0; k < NF; k = k+1)
                    if ($signed(feats[(p*NF + k)*IW +: IW]) !== $signed(e[k])) err = err + 1;
            end
            @(negedge clk);
        end
        $fclose(fd_iq); $fclose(fd_ft);
        $display("nn_dpd_feat_par (P=%0d): mismatches = %0d", P, err);
        if (err == 0) $display("PASS: parallel (multi-lane) feature builder matches Python");
        else          $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

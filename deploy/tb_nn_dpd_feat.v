// ============================================================================
// tb_nn_dpd_feat.v - checks nn_dpd_feat (delay line + integer |x|) vs Python
// golden vectors (deploy/iq.hex input, deploy/feat.hex expected features).
// ============================================================================
`timescale 1ns/1ps
`default_nettype none
module tb_nn_dpd_feat;
    localparam M = 4, IW = 16, NV = 64, NF = 3*M;
    reg clk = 0, rst = 1, ce = 0;
    reg  signed [IW-1:0] xi = 0, xq = 0;
    wire [NF*IW-1:0] feats;
    integer i, k, fd_iq, fd_ft, rc, err = 0;
    reg [15:0] a, b, e [0:NF-1];

    nn_dpd_feat #(.M(M), .IW(IW)) dut (
        .clk(clk), .rst(rst), .ce(ce), .xi(xi), .xq(xq), .feats(feats));

    always #5 clk = ~clk;

    initial begin
        fd_iq = $fopen("iq.hex", "r");
        fd_ft = $fopen("feat.hex", "r");
        if (fd_iq == 0 || fd_ft == 0) begin $display("ERROR: iq.hex/feat.hex missing"); $finish; end
        rst = 1; ce = 0; repeat (3) @(negedge clk); rst = 0; ce = 1;
        @(negedge clk);
        for (i = 0; i < NV; i = i+1) begin
            rc = $fscanf(fd_iq, "%h %h\n", a, b);
            xi = a[IW-1:0]; xq = b[IW-1:0];
            for (k = 0; k < NF; k = k+1) rc = $fscanf(fd_ft, "%h", e[k]);
            #1;
            if (i < 2) $display("i=%0d exp[0..2]=%04x %04x %04x got=%04x %04x %04x", i,
                               e[0],e[1],e[2], feats[15:0],feats[31:16],feats[47:32]);
            for (k = 0; k < NF; k = k+1)
                if ($signed(feats[k*IW +: IW]) !== $signed(e[k])) err = err + 1;
            @(negedge clk);
        end
        $fclose(fd_iq); $fclose(fd_ft);
        $display("nn_dpd_feat: mismatches = %0d", err);
        if (err == 0) $display("PASS: feature builder (delay line + isqrt) matches Python");
        else          $display("FAIL");
        $finish;
    end
endmodule
`default_nettype wire

// ============================================================================
// nn_dpd_scale.v - feature->activation scaling for the MLP input.
//   fout[i] = saturate( (fin[i] * KQ) >>> KSH )     (int16, K = 32700/max|feat|)
// ============================================================================
`default_nettype none
`include "par_params.vh"
module nn_dpd_scale #(
    parameter NF = 24            // P * 3 * M
)(
    input  wire [NF*16-1:0] fin,
    output wire [NF*16-1:0] fout
);
    genvar i;
    generate
        for (i=0; i<NF; i=i+1) begin : gs
            wire signed [15:0] x = fin[i*16 +: 16];
            wire signed [47:0] p = x * `PAR_KQ;
            wire signed [47:0] s = p >>> `PAR_KSH;
            wire signed [15:0] y = (s > 32767) ? 16'sd32767 : (s < -32768) ? -16'sd32768 : s[15:0];
            assign fout[i*16 +: 16] = y;
        end
    endgenerate
endmodule
`default_nettype wire

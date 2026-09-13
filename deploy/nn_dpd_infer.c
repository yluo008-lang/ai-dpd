/* nn_dpd_infer.c - fixed-point inference of the exported RVTDNN DPD.
 *
 * Reproduces the bit-true integer datapath of quant_export.py / nn_dpd_layer.v:
 *   acc = sum W*x                              (int64)
 *   out = (acc*REQ + (BQ<<SHIFT) + (1<<(SHIFT-1))) >> SHIFT   (int16, ReLU)
 * Output layer is kept in real units (feeds the residual add).
 *
 *   build: gcc -O2 -I. -o nn_dpd_infer nn_dpd_infer.c -lm
 *   run  : ./nn_dpd_infer     (from deploy/)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "weights.h"

static void dense(const short *W, const int *BQ, int nout, int nin,
                  const short *x, int req, int sh, int relu, short *y){
    for (int o = 0; o < nout; ++o){
        long long acc = 0;
        for (int i = 0; i < nin; ++i)
            acc += (long long)W[o*nin + i] * (long long)x[i];
        long long t = acc*(long long)req + ((long long)BQ[o] << sh) + (1LL << (sh-1));
        long long v = t >> sh;                       /* arithmetic shift */
        if (relu && v < 0) v = 0;
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        y[o] = (short)v;
    }
}

int main(void){
    FILE *f = fopen("ref_io.txt","r");
    if (!f){ perror("ref_io.txt"); return 1; }
    short a0[NNE_IN], a1[NNE_MID], a2[NNE_MID];
    double feats[NNE_IN], er, ei;
    int n = 0; double maxerr = 0.0;
    printf("nn_dpd_infer: %d -> %d -> 2 (int16, integer requant)\n", NNE_IN, NNE_MID);
    while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                  &feats[0],&feats[1],&feats[2],&feats[3],&feats[4],&feats[5],
                  &feats[6],&feats[7],&feats[8],&feats[9],&feats[10],&feats[11],
                  &er,&ei) == 14){
        for (int i = 0; i < NNE_IN; ++i){
            double r = nearbyint(feats[i]/SX0);
            if (r > 32767) r = 32767; if (r < -32768) r = -32768;
            a0[i] = (short)r;
        }
        dense(W0q, BQ0q, NNE_MID, NNE_IN, a0, REQ0, SHIFTQ, 1, a1);
        dense(W1q, BQ1q, NNE_MID, NNE_MID, a1, REQ1, SHIFTQ, 1, a2);
        long aa = 0, bb = 0;
        for (int i = 0; i < NNE_MID; ++i){
            aa += (long)W2q[0*NNE_MID + i]*(long)a2[i];
            bb += (long)W2q[1*NNE_MID + i]*(long)a2[i];
        }
        double outr = aa*(SW2*SX2) + b_out[0] + feats[0];   /* + residual x(n) */
        double outi = bb*(SW2*SX2) + b_out[1] + feats[1];
        double e = hypot(outr-er, outi-ei);
        if (e > maxerr) maxerr = e;
        if (n < 4) printf("  n=%d  C=(%+.6f,%+.6f)  ref=(%+.6f,%+.6f)\n", n, outr, outi, er, ei);
        n++;
    }
    fclose(f);
    printf("checked %d samples, max |C - ref| = %.3e\n", n, maxerr);
    printf(maxerr < 1e-5 ? "PASS: C inference matches reference\n" : "FAIL: mismatch\n");
    return 0;
}

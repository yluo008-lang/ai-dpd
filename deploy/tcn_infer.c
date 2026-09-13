/* tcn_infer.c - int16 inference of the exported TCN DPD (conv1d + ReLU + linear).
 * Conv layout: x (Cin,N) = x[c*N+t]; W (Cout,Cin,3) = W[o*Cin*3 + c*3 + j].
 * Mirrors quant_tcn_export.py. Reads deploy/tcn_ref_io.txt and self-checks.
 *
 *   gcc -O2 -I. -o tcn_infer tcn_infer.c -lm && ./tcn_infer
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tcn_weights.h"

#define NT 32
static short SAT(long v){ return v>32767?32767:(v<-32768?-32768:(short)v); }

static void conv(const short *W, const double *B, int Cout, int Cin, int D, int N,
                 const short *x, double sw, double sx, double sy, short *y){
    for (int o=0; o<Cout; ++o)
        for (int t=0; t<N; ++t){
            long acc = 0;
            for (int c=0; c<Cin; ++c)
                for (int j=0; j<3; ++j){
                    int idx = t - D*(2-j);
                    if (idx >= 0) acc += (long)W[o*Cin*3 + c*3 + j] * (long)x[c*N + idx];
                }
            double real = acc*(sw*sx) + B[o];
            if (real < 0) real = 0;
            y[o*N + t] = SAT((long)nearbyint(real/sy));
        }
}

int main(void){
    static short x0[2*NT]; static short a1[32*NT], a2[32*NT], a3[32*NT];
    int xi[NT], xq[NT], er[NT], ei[NT];
    FILE *f = fopen("tcn_ref_io.txt","r");
    if (!f){ perror("tcn_ref_io.txt"); return 1; }
    for (int t=0;t<NT;t++){
        char a[8],b[8],c[8],d[8];
        if (fscanf(f,"%s %s %s %s",a,b,c,d)!=4) return 1;
        x0[0*NT+t] = (short)strtol(a,NULL,16);
        x0[1*NT+t] = (short)strtol(b,NULL,16);
        er[t] = (short)strtol(c,NULL,16); ei[t] = (short)strtol(d,NULL,16);
    }
    fclose(f);
    short xs[2*NT];                                 /* input activations scaled by SX0 */
    for (int i=0;i<2*NT;i++) xs[i] = SAT((long)nearbyint(x0[i]/(32768.0*TCN_SX0)));
    conv(TCN_W0, TCN_B0, 32, 2,  TCN_D0, NT, xs, TCN_SW0, TCN_SX0, TCN_SY0, a1);
    conv(TCN_W1, TCN_B1, 32, 32, TCN_D1, NT, a1, TCN_SW1, TCN_SY0, TCN_SY1, a2);
    conv(TCN_W2, TCN_B2, 32, 32, TCN_D2, NT, a2, TCN_SW2, TCN_SY1, TCN_SY2, a3);
    printf("DBG C a1=%d %d %d %d | a2=%d %d %d %d | a3=%d %d %d %d\n",
           a1[0*NT+0],a1[1*NT+0],a1[2*NT+0],a1[3*NT+0],
           a2[0*NT+0],a2[1*NT+0],a2[2*NT+0],a2[3*NT+0],
           a3[0*NT+0],a3[1*NT+0],a3[2*NT+0],a3[3*NT+0]);
    int nerr=0; double maxe=0;
    for (int t=0;t<NT;t++){
        double hr=0, hi=0;
        for (int c=0;c<32;c++){
            hr += (TCN_WH[0*32+c]*TCN_SWH) * (a3[c*NT+t]*TCN_SY2);
            hi += (TCN_WH[1*32+c]*TCN_SWH) * (a3[c*NT+t]*TCN_SY2);
        }
        hr += TCN_BH[0]; hi += TCN_BH[1];
        double o_r = hr + x0[0*NT+t]/32768.0;        /* + residual x(n) */
        double o_i = hi + x0[1*NT+t]/32768.0;
        double e = hypot(o_r*32768 - er[t], o_i*32768 - ei[t]);
        if (e > maxe) maxe = e;
        if (e > 1.0) nerr++;
    }
    printf("TCN int16 inference: layers 2->32->32->32->2 (dil 1,2,4)\n");
    printf("checked %d samples, max |C-ref| = %.1f LSB, samples off >1 LSB: %d\n", NT, maxe, nerr);
    printf(nerr==0 ? "PASS: C conv inference matches reference\n" : "FAIL\n");
    return 0;
}

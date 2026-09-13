/* hybrid_infer.c - int16 inference of the Hybrid DPD (MP-polynomial + neural).
 *
 *   D(x) = W_mp . phi_MP(x) + MLP([I,Q,|x|] taps)
 *   phi_MP basis (col k = m*Np+j): x(n-m)*|x(n-m)|^(2j), iterated as Q1.15.
 * Reads deploy/hybrid_ref_io.txt (input Q15 + expected Q15) and self-checks.
 *
 *   gcc -O2 -I. -o hybrid_infer hybrid_infer.c -lm && ./hybrid_infer
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hybrid_weights.h"

#define NPTS 32
static short SAT(long v){ return v>32767?32767:(v<-32768?-32768:(short)v); }
static long isqrtL(long v){ long r=(long)sqrt((double)(v>0?v:0)); while((r+1)*(r+1)<=(long)v) r++; while(r*r>(long)v) r--; return r; }

static void dense(const short *W, const int *BQ, int nout, int nin, const short *x,
                  int req, int relu, short *y){
    for (int o=0;o<nout;o++){
        long long acc=0;
        for (int i=0;i<nin;i++) acc += (long long)W[o*nin+i]*(long long)x[i];
        long long t = acc*(long long)req + ((long long)BQ[o] << HY_SHIFT) + (1LL << (HY_SHIFT-1));
        long long v = t >> HY_SHIFT;
        if (relu && v<0) v=0;
        if (v>32767) v=32767; if (v<-32768) v=-32768;
        y[o]=(short)v;
    }
}

int main(void){
    static short xi[NPTS], xq[NPTS], a1[HY_MID], a2[HY_MID], fq[3*HY_M];
    int er[NPTS], ei[NPTS];
    FILE *f = fopen("hybrid_ref_io.txt","r");
    if (!f){ perror("hybrid_ref_io.txt"); return 1; }
    for (int t=0;t<NPTS;t++){
        char a[8],b[8],c[8],d[8];
        if (fscanf(f,"%s %s %s %s",a,b,c,d)!=4) return 1;
        xi[t]=(short)strtol(a,NULL,16); xq[t]=(short)strtol(b,NULL,16);
        er[t]=(short)strtol(c,NULL,16); ei[t]=(short)strtol(d,NULL,16);
    }
    fclose(f);

    int nerr=0; double maxe=0;
    for (int t=0;t<NPTS;t++){
        /* --- polynomial branch: build phi (Q15) and MAC with W_mp --- */
        double lin_r=0, lin_i=0;
        for (int m=0;m<HY_M;m++){
            int idx = t-m;
            long I = (idx>=0)? xi[idx] : 0, Q = (idx>=0)? xq[idx] : 0;
            long mag2 = I*I + Q*Q;
            long tr = I, ti = Q;
            for (int j=0;j<HY_NP;j++){
                int k = m*HY_NP + j;
                short pre = SAT((long)nearbyint((double)tr/(32768.0*HY_Sphi)));   /* phi.re */
                short pim = SAT((long)nearbyint((double)ti/(32768.0*HY_Sphi)));   /* phi.im */
                int row0 = 0*2*HY_NC, row1 = 1*2*HY_NC;
                lin_r += (HY_WL[row0 + k]*HY_Sl) * (pre*HY_Sphi);
                lin_i += (HY_WL[row1 + k]*HY_Sl) * (pre*HY_Sphi);
                lin_r += (HY_WL[row0 + HY_NC + k]*HY_Sl) * (pim*HY_Sphi);
                lin_i += (HY_WL[row1 + HY_NC + k]*HY_Sl) * (pim*HY_Sphi);
                tr = (tr*mag2) >> 30; ti = (ti*mag2) >> 30;
            }
        }
        lin_r += HY_BL[0]; lin_i += HY_BL[1];
        /* --- neural branch: features [I,Q,|x|] per tap --- */
        for (int m=0;m<HY_M;m++){
            int idx = t-m;
            long I = (idx>=0)? xi[idx] : 0, Q = (idx>=0)? xq[idx] : 0;
            long mag = isqrtL(I*I + Q*Q);
            fq[3*m+0] = SAT((long)nearbyint((double)I/32768.0/HY_SX0));
            fq[3*m+1] = SAT((long)nearbyint((double)Q/32768.0/HY_SX0));
            fq[3*m+2] = SAT((long)nearbyint((double)mag/32768.0/HY_SX0));
        }
        dense(HY_W0, HY_B0, HY_MID, HY_IN, fq, HY_REQ0, 1, a1);
        dense(HY_W2, HY_B2, HY_MID, HY_MID, a1, HY_REQ1, 1, a2);
        double nn_r=0, nn_i=0;
        for (int c=0;c<HY_MID;c++){
            nn_r += (HY_W4[0*HY_MID+c]*HY_SW4)*(a2[c]*HY_SX2);
            nn_i += (HY_W4[1*HY_MID+c]*HY_SW4)*(a2[c]*HY_SX2);
        }
        nn_r += HY_B4[0]; nn_i += HY_B4[1];
        double o_r = lin_r + nn_r, o_i = lin_i + nn_i;
        double e = hypot(o_r*32768 - er[t], o_i*32768 - ei[t]);
        if (e>16.0) nerr++;
        if (e>maxe) maxe=e;
    }
    printf("hybrid_infer: MP(%d terms) + MLP(%d) (int16)\n", HY_NC, HY_MID);
    printf("checked %d samples, max |C-ref| = %.1f LSB (tolerance 16), samples off >16 LSB: %d\n", NPTS, maxe, nerr);
    printf(nerr==0 ? "PASS: C hybrid inference matches reference\n" : "FAIL\n");
    return 0;
}

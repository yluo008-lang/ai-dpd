"""run_size.py - does the sequence/FFT length N matter?

Tests N in {4096, 16384, 65536} for:
  - PolyMP (LS regression matrix: N rows x 12 cols -> already overdetermined)
  - DLA-RVTDNN (neural: training-set size & gradient noise)
Report ACLR mean +- std over 3 seeds.
"""
import numpy as np, torch, time, dsp, pa
from models import RVTDNN
from train import train_nn, train_dla, nn_apply, poly_dpd_train, poly_dpd_apply

torch.manual_seed(0)
TS, EP, SEEDS = 999, 1500, (11, 22, 33)

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

if __name__ == "__main__":
    print(f"{'N':>7s} {'method':>8s} {'ACLR(dB)':>17s} {'EVM(%)':>9s}  {'sec/train':>9s}")
    for N in (4096, 16384, 65536):
        for method in ("poly", "DLA"):
            res, t0 = [], time.time()
            for seed in SEEDS:
                xtr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
                xte = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
                PA = pa.MP_PA(); G = PA.c[0]
                if method == "poly":
                    c = poly_dpd_train(PA(xtr)/G, xtr); y = PA(poly_dpd_apply(xte, c))
                else:
                    m = RVTDNN(4, 32, "relu")
                    train_dla(m, xtr, pa.torch_mp_pa(), G, epochs=EP, lr=1e-2)
                    y = PA(nn_apply(m, xte))
                res.append(met(xte, y))
            a = np.array([r[0] for r in res]); e = np.array([r[1] for r in res])
            dt = (time.time()-t0)/len(SEEDS)
            print(f"{N:7d} {method:>8s} {a.mean():8.2f} +- {a.std():4.2f}   {e.mean():7.3f}   {dt:8.1f}")

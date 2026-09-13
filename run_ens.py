"""run_ens.py - variance reduction: residual zero-init + ensemble.

Zero-init the residual branch (D(x)=x at start), then average the corrections of
K independently-initialised models. Report ACLR/EVM mean +- std across 5 seeds
for single model vs K=5 ensemble, ILA and DLA.
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN
from train import train_nn, train_dla, nn_apply

torch.manual_seed(0)
N, EP, TS = 8192, 1500, 999
SEEDS = [11, 22, 33, 44, 55]

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def train_one(xtr, PA, G, tseed, method):
    torch.manual_seed(tseed)                       # varied init per ensemble member
    m = RVTDNN(4, 32, "relu")
    if method == "DLA":
        train_dla(m, xtr, pa.torch_mp_pa(), G, epochs=EP, lr=1e-2)
    else:
        train_nn(m, PA(xtr)/G, xtr, epochs=EP, lr=1e-2)
    return m

if __name__ == "__main__":
    PA = pa.MP_PA(); G = PA.c[0]
    print(f"{'method':5s} {'K':>2s} {'ACLR(dB)':>17s} {'EVM(%)':>12s}")
    for method in ("ILA", "DLA"):
        for K in (1, 5):
            res = []
            for si, seed in enumerate(SEEDS):
                xtr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
                xte = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
                ms = [train_one(xtr, PA, G, 1000 + si*10 + i, method) for i in range(K)]
                zc = np.mean([nn_apply(m, xte) for m in ms], axis=0)   # average correction
                res.append(met(xte, PA(zc)))
            a = np.array([r[0] for r in res]); e = np.array([r[1] for r in res])
            print(f"{method:5s} {K:2d} {a.mean():8.2f} +- {a.std():4.2f}   {e.mean():6.3f} +- {e.std():.3f}")

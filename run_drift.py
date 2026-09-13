"""run_drift.py - PA DRIFT robustness (train/test PA params differ).

The earlier mismatch experiment used the SAME I/Q-imbalance for train and test.
Here the PA changes between training and evaluation (beta_train != beta_test), so
we can show how PolyMP vs NN-DPD degrade under PA drift.
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN
from train import train_dla, nn_apply, poly_dpd_train, poly_dpd_apply

torch.manual_seed(0)
N, EP, TS = 4096, 1500, 999
SEEDS = [11, 22, 33]

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def trial(beta_tr, beta_te, seed):
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
    PAtr = pa.IQMP_PA(beta=beta_tr); PAte = pa.IQMP_PA(beta=beta_te)
    G = PAtr.c[0]
    out = {}
    out["no-DPD"] = met(x_te, PAte(x_te))
    c = poly_dpd_train(PAtr(x_tr)/G, x_tr)                # trained on PAtr
    out["PolyMP"] = met(x_te, PAte(poly_dpd_apply(x_te, c)))   # applied to PAte
    m = RVTDNN(4, 32, "relu")
    train_dla(m, x_tr, pa.torch_iqmp_pa(beta=beta_tr), G, epochs=EP, lr=1e-2)
    out["DLA-NN"] = met(x_te, PAte(nn_apply(m, x_te)))
    return out

if __name__ == "__main__":
    for btr, bte in [(0.05, 0.05), (0.05, 0.08), (0.08, 0.05), (0.05, 0.12)]:
        rows = [trial(btr, bte, s) for s in SEEDS]
        tag = "same" if btr == bte else "DRIFT"
        print(f"\n--- beta_train={btr}  beta_test={bte}  [{tag}] ---")
        for k in rows[0]:
            a = np.array([r[k][0] for r in rows]); e = np.array([r[k][1] for r in rows])
            print(f"  {k:8s} ACLR {a.mean():7.2f} +- {a.std():4.2f}   EVM {e.mean():6.3f}%")

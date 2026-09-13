"""run_hybrid.py - hybrid DPD: MP polynomial (exact inverse) + neural residual.

D(x) = W_mp . phi_MP(x) + NN(x), trained end-to-end (DLA through a differentiable
PA). Tests whether embedding the polynomial prior closes the ~10 dB gap to PolyMP
while keeping the NN's flexibility.
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN, HybridDPD
from train import train_dla, nn_apply, poly_dpd_train, poly_dpd_apply

torch.manual_seed(0)
N, EP, TS = 8192, 2000, 999
SEEDS = [11, 22, 33]

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

if __name__ == "__main__":
    print(f"{'method':12s} {'ACLR(dB)':>17s} {'EVM(%)':>13s}")
    for name in ("PolyMP", "RVTDNN-DLA", "Hybrid-DLA"):
        res = []
        for seed in SEEDS:
            xtr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
            xte = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
            PA = pa.MP_PA(); G = PA.c[0]
            if name == "PolyMP":
                c = poly_dpd_train(PA(xtr)/G, xtr); y = PA(poly_dpd_apply(xte, c))
            else:
                m = HybridDPD(4, 3, 32) if "Hybrid" in name else RVTDNN(4, 32, "relu")
                if "Hybrid" in name:
                    c = poly_dpd_train(PA(xtr)/G, xtr)          # LS poly inverse as prior
                    Nc = len(c)
                    with torch.no_grad():
                        m.lin.weight[0, :Nc] = torch.tensor(c.real); m.lin.weight[0, Nc:] = torch.tensor(-c.imag)
                        m.lin.weight[1, :Nc] = torch.tensor(c.imag); m.lin.weight[1, Nc:] = torch.tensor(c.real)
                        m.lin.bias.zero_()
                train_dla(m, xtr, pa.torch_mp_pa(), G, epochs=EP, lr=(5e-3 if "Hybrid" in name else 1e-2))
                y = PA(nn_apply(m, xte))
            res.append(met(xte, y))
        a = np.array([r[0] for r in res]); e = np.array([r[1] for r in res])
        print(f"{name:12s} {a.mean():8.2f} +- {a.std():4.2f}   {e.mean():6.3f} +- {e.std():.3f}")

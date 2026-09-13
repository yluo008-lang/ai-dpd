"""run_hybrid_quant.py - Hybrid int16 quant: per-tensor vs PER-COLUMN MP basis.

The MP basis terms span ~50x in magnitude, so a single basis scale ruins the small
terms (the poly branch must cancel to -70 dB). Give each basis column its own scale.
Reports the (honest) CORRECTION fidelity and end-to-end ACLR/EVM loss.
"""
import numpy as np, torch, dsp, pa
from models import HybridDPD, real_features
from train import nn_apply

M, Np, H, SHIFT = 4, 3, 32, 30
def q(a, s): return np.clip(np.round(np.asarray(a)/s), -32768, 32767).astype(np.int64)
def req(acc, reqk, bq, sh):
    t = acc*reqk + (bq << sh) + (1 << (sh-1)); return t >> sh
def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def main():
    x_tr = dsp.scale(dsp.gen_ofdm(4096, seed=11), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(4096, seed=999), 0.25)
    PA = pa.MP_PA()
    m = HybridDPD(M, Np, H); m.load_state_dict(torch.load("Hybrid.pt"))
    sd = m.state_dict()
    Wl = sd["lin.weight"].numpy().astype(np.float64); bl = sd["lin.bias"].numpy().astype(np.float64)
    W0 = sd["net.0.weight"].numpy().astype(np.float64); b0 = sd["net.0.bias"].numpy().astype(np.float64)
    W2 = sd["net.2.weight"].numpy().astype(np.float64); b2 = sd["net.2.bias"].numpy().astype(np.float64)
    W4 = sd["net.4.weight"].numpy().astype(np.float64); b4 = sd["net.4.bias"].numpy().astype(np.float64)
    zf = nn_apply(m, x_te); af = met(x_te, PA(zf))
    print(f"float Hybrid : ACLR {af[0]:7.2f}  EVM {af[1]:.4f}%")

    # MLP branch scales (from train)
    f0 = real_features(torch.as_tensor(x_tr, dtype=torch.complex64), M).numpy().astype(np.float64)
    h1 = np.maximum(f0 @ W0.T + b0, 0); h2 = np.maximum(h1 @ W2.T + b2, 0)
    SX0 = float(np.abs(f0).max())/32700.0
    SX1 = float(np.abs(h1).max())/32700.0; SX2 = float(np.abs(h2).max())/32700.0
    SW0 = float(np.abs(W0).max())/32767.0; SW2 = float(np.abs(W2).max())/32767.0; SW4 = float(np.abs(W4).max())/32767.0
    REQ0 = int(round(SW0*SX0/SX1*(1 << SHIFT))); BQ0 = np.round(b0/SX1).astype(np.int64)
    REQ1 = int(round(SW2*SX1/SX2*(1 << SHIFT))); BQ1 = np.round(b2/SX2).astype(np.int64)

    def nn_branch(x):
        fe = real_features(torch.as_tensor(x, dtype=torch.complex64), M).numpy().astype(np.float64)
        a0 = q(fe, SX0)
        a1 = np.clip(np.maximum(req(a0 @ q(W0, SW0).T, REQ0, BQ0, SHIFT), 0), -32768, 32767)
        a2 = np.clip(np.maximum(req(a1 @ q(W2, SW2).T, REQ1, BQ1, SHIFT), 0), -32768, 32767)
        return (a2 @ q(W4, SW4).T)*(SW4*SX2) + b4

    # basis scales from train
    phi_tr = pa.mp_basis(x_tr, M, Np); f_tr = np.concatenate([phi_tr.real, phi_tr.imag], 1)
    Sphi_t = float(np.abs(f_tr).max())/32700.0
    Sp_col = np.abs(f_tr).max(axis=0)/32700.0                    # (2Nc,)
    Sl = float(np.abs(Wl).max())/32767.0

    def poly_branch(x, per_col):
        phi = pa.mp_basis(x, M, Np); f = np.concatenate([phi.real, phi.imag], 1)
        if per_col:
            fq = np.stack([q(f[:, j], Sp_col[j]) for j in range(f.shape[1])], 1)
            yl = np.zeros((len(x), 2))
            for j in range(f.shape[1]):
                yl[:, 0] += Wl[0, j]*Sp_col[j]*fq[:, j]
                yl[:, 1] += Wl[1, j]*Sp_col[j]*fq[:, j]
        else:
            fq = q(f, Sphi_t)
            yl = (fq @ q(Wl, Sl).T)*(Sl*Sphi_t)
        return yl + bl

    for name, pc in (("per-tensor basis", False), ("PER-COLUMN basis", True)):
        zq = (poly_branch(x_te, pc)[:, 0] + nn_branch(x_te)[:, 0]) + 1j*(poly_branch(x_te, pc)[:, 1] + nn_branch(x_te)[:, 1])
        zq_tr = (poly_branch(x_tr, pc)[:, 0] + nn_branch(x_tr)[:, 0]) + 1j*(poly_branch(x_tr, pc)[:, 1] + nn_branch(x_tr)[:, 1])
        zf_tr = nn_apply(m, x_tr)
        cfid = dsp.nmse_db(zf_tr - x_tr, zq_tr - x_tr)           # honest: correction-only
        a = met(x_te, PA(zq))
        print(f"{name:16s}: corr fid {-cfid:6.1f} dB | int16 ACLR {a[0]:7.2f} ({a[0]-af[0]:+.2f})  EVM {a[1]:.4f}%")

if __name__ == "__main__":
    main()

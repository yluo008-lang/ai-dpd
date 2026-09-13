"""run_hybrid_bits.py - how many bits does the polynomial branch need?

Isolation showed the +21 dB Hybrid fixed-point loss is entirely the MP-polynomial
branch (NN branch is lossless at int16). Sweep the poly-branch bit width.
"""
import numpy as np, torch, dsp, pa
from models import HybridDPD, real_features
from train import nn_apply

M, Np, H = 4, 3, 32
def qb(a, s, bits):
    qmax = 2**(bits-1) - 1
    return np.clip(np.round(np.asarray(a)/s), -qmax-1, qmax).astype(np.int64)
def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def main():
    x_tr = dsp.scale(dsp.gen_ofdm(4096, seed=11), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(4096, seed=999), 0.25)
    PA = pa.MP_PA()
    m = HybridDPD(M, Np, H); m.load_state_dict(torch.load("Hybrid.pt"))
    sd = m.state_dict()
    Wl, bl = sd["lin.weight"].numpy().astype(np.float64), sd["lin.bias"].numpy().astype(np.float64)
    W0, b0 = sd["net.0.weight"].numpy().astype(np.float64), sd["net.0.bias"].numpy().astype(np.float64)
    W2, b2 = sd["net.2.weight"].numpy().astype(np.float64), sd["net.2.bias"].numpy().astype(np.float64)
    W4, b4 = sd["net.4.weight"].numpy().astype(np.float64), sd["net.4.bias"].numpy().astype(np.float64)
    zf = nn_apply(m, x_te); zf_tr = nn_apply(m, x_tr); af = met(x_te, PA(zf))
    print(f"float torch : ACLR {af[0]:7.2f}  EVM {af[1]:.4f}%")
    phi_tr = pa.mp_basis(x_tr, M, Np); f_tr = np.concatenate([phi_tr.real, phi_tr.imag], 1)

    def nn_float(x):
        fe = real_features(torch.as_tensor(x, dtype=torch.complex64), M).numpy().astype(np.float64)
        h = np.maximum(fe @ W0.T + b0, 0); h = np.maximum(h @ W2.T + b2, 0)
        return h @ W4.T + b4

    def to_cplx(yl, nn):
        return (yl[:, 0] + nn[:, 0]) + 1j*(yl[:, 1] + nn[:, 1])

    def poly(x, bits, per_col):
        phi = pa.mp_basis(x, M, Np); f = np.concatenate([phi.real, phi.imag], 1)
        if per_col:
            Sp = np.abs(f_tr).max(0)/(2**(bits-1)-1)
            fq = np.stack([qb(f[:, j], Sp[j], bits) for j in range(f.shape[1])], 1)
            yl = np.zeros((len(x), 2))
            for j in range(f.shape[1]):
                yl[:, 0] += Wl[0, j]*Sp[j]*fq[:, j]; yl[:, 1] += Wl[1, j]*Sp[j]*fq[:, j]
            return yl + bl
        Sp = float(np.abs(f_tr).max())/(2**(bits-1)-1)
        fq = qb(f, Sp, bits)
        return (fq * Sp) @ Wl.T + bl

    for bits in (16, 20, 24, 32, 40):
        for pc in (False, True):
            zq = to_cplx(poly(x_te, bits, pc), nn_float(x_te))
            zq_tr = to_cplx(poly(x_tr, bits, pc), nn_float(x_tr))
            a = met(x_te, PA(zq))
            fid = -dsp.nmse_db(zf_tr - x_tr, zq_tr - x_tr)
            tag = "per-col" if pc else "per-ten"
            print(f"poly {bits:2d}b {tag}: fid {fid:6.1f} dB | ACLR {a[0]:7.2f} ({a[0]-af[0]:+.2f})  EVM {a[1]:.4f}%")

if __name__ == "__main__":
    main()

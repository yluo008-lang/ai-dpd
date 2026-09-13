"""run_hybrid_isolate.py - isolate the Hybrid fixed-point loss.

(0) bug check : numpy FLOAT forward vs torch float   (should match; if not -> bug)
(1) poly-only int16 (NN float)
(2) NN-only int16   (poly float)
(3) both int16
Report honest correction-only fidelity + end-to-end ACLR/EVM for each.
"""
import numpy as np, torch, dsp, pa
from models import HybridDPD, real_features
from train import nn_apply

M, Np, H, SHIFT = 4, 3, 32, 30
def q(a, s): return np.clip(np.round(np.asarray(a)/s), -32768, 32767).astype(np.int64)
def req(acc, rk, bq, sh):
    t = acc*rk + (bq << sh) + (1 << (sh-1)); return t >> sh
def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def load():
    m = HybridDPD(M, Np, H); m.load_state_dict(torch.load("Hybrid.pt"))
    sd = m.state_dict()
    return m, {k: sd[k].numpy().astype(np.float64) for k in sd}

def main():
    x_tr = dsp.scale(dsp.gen_ofdm(4096, seed=11), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(4096, seed=999), 0.25)
    PA = pa.MP_PA()
    m, W = load()
    Wl, bl = W["lin.weight"], W["lin.bias"]
    W0, b0 = W["net.0.weight"], W["net.0.bias"]
    W2, b2 = W["net.2.weight"], W["net.2.bias"]
    W4, b4 = W["net.4.weight"], W["net.4.bias"]
    zf = nn_apply(m, x_te); af = met(x_te, PA(zf))
    zf_tr = nn_apply(m, x_tr)
    print(f"float torch  : ACLR {af[0]:7.2f}  EVM {af[1]:.4f}%")

    # scales from train
    phi_tr = pa.mp_basis(x_tr, M, Np); f_tr = np.concatenate([phi_tr.real, phi_tr.imag], 1)
    Sphi = float(np.abs(f_tr).max())/32700.0; Sl = float(np.abs(Wl).max())/32767.0
    fe_tr = real_features(torch.as_tensor(x_tr, dtype=torch.complex64), M).numpy().astype(np.float64)
    h1 = np.maximum(fe_tr @ W0.T + b0, 0); h2 = np.maximum(h1 @ W2.T + b2, 0)
    SX0 = float(np.abs(fe_tr).max())/32700.0; SX1 = float(np.abs(h1).max())/32700.0; SX2 = float(np.abs(h2).max())/32700.0
    SW0 = float(np.abs(W0).max())/32767.0; SW2 = float(np.abs(W2).max())/32767.0; SW4 = float(np.abs(W4).max())/32767.0
    REQ0 = int(round(SW0*SX0/SX1*(1 << SHIFT))); BQ0 = np.round(b0/SX1).astype(np.int64)
    REQ1 = int(round(SW2*SX1/SX2*(1 << SHIFT))); BQ1 = np.round(b2/SX2).astype(np.int64)

    def poly(x, quant):
        phi = pa.mp_basis(x, M, Np); f = np.concatenate([phi.real, phi.imag], 1)
        if not quant:
            return f @ Wl.T + bl
        fq = q(f, Sphi)
        return (fq @ q(Wl, Sl).T)*(Sl*Sphi) + bl

    def nn(x, quant):
        fe = real_features(torch.as_tensor(x, dtype=torch.complex64), M).numpy().astype(np.float64)
        if not quant:
            y = np.maximum(fe @ W0.T + b0, 0); y = np.maximum(y @ W2.T + b2, 0)
            return y @ W4.T + b4
        a0 = q(fe, SX0)
        a1 = np.clip(np.maximum(req(a0 @ q(W0, SW0).T, REQ0, BQ0, SHIFT), 0), -32768, 32767)
        a2 = np.clip(np.maximum(req(a1 @ q(W2, SW2).T, REQ1, BQ1, SHIFT), 0), -32768, 32767)
        return (a2 @ q(W4, SW4).T)*(SW4*SX2) + b4

    def comb(x, qp, qn):
        pl = poly(x, qp); nl = nn(x, qn)
        return (pl[:, 0]+nl[:, 0]) + 1j*(pl[:, 1]+nl[:, 1])

    # (0) bug check: numpy float vs torch float
    z_nf = comb(x_te, False, False)
    print(f"numpy FLOAT  : ACLR {met(x_te, PA(z_nf))[0]:7.2f}  | torch-vs-numpy fid {-dsp.nmse_db(zf-x_te, z_nf-x_te):.1f} dB")

    for name, qp, qn in (("poly int16 only",  True,  False),
                         ("NN int16 only",    False, True),
                         ("both int16",       True,  True)):
        zq = comb(x_te, qp, qn); zq_tr = comb(x_tr, qp, qn)
        a = met(x_te, PA(zq))
        fid = -dsp.nmse_db(zf_tr - x_tr, zq_tr - x_tr)
        print(f"{name:16s}: corr fid {fid:6.1f} dB | int16 ACLR {a[0]:7.2f} ({a[0]-af[0]:+.2f})  EVM {a[1]:.4f}%")

if __name__ == "__main__":
    main()

"""run_mismatch.py - matched (MP) vs MISMATCHED (I/Q imbalance) PA.

Tests the actual premise: an MP/GMP DPD is the exact inverse of an MP PA (so poly
wins by construction); the NN's value can only show when the PA has structure a
memory polynomial cannot invert -- here an image term b*conj(.).
Also: TCN now has a residual skip; paired stats; end-to-end pct sweep.
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN, TCN
from train import train_nn, train_dla, nn_apply, poly_dpd_train, poly_dpd_apply

torch.manual_seed(0)
N, EP, TS = 4096, 1500, 999
SEEDS = [11, 22, 33, 44, 55]

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

def run(PA, pat, seed):
    xtr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
    xte = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
    G = PA.c[0]; ytr = PA(xtr); o = {}
    o["no-DPD"] = met(xte, PA(xte))
    c = poly_dpd_train(ytr/G, xtr); o["PolyMP"] = met(xte, PA(poly_dpd_apply(xte, c)))
    mi = RVTDNN(4, 32, "relu"); train_nn(mi, ytr/G, xtr, epochs=EP, lr=1e-2)
    o["ILA-NN"] = met(xte, PA(nn_apply(mi, xte)))
    md = RVTDNN(4, 32, "relu"); train_dla(md, xtr, pat, G, epochs=EP, lr=1e-2)
    o["DLA-NN"] = met(xte, PA(nn_apply(md, xte)))
    mt = TCN(4, 16); train_nn(mt, ytr/G, xtr, epochs=EP, lr=3e-3)
    o["TCN"] = met(xte, PA(nn_apply(mt, xte)))
    return o

def summarize(name, rows):
    print(f"\n=== {name}  (mean +- std, n={len(rows)}) ===")
    print(f"  {'method':10s} {'ACLR(dB)':>16s} {'EVM(%)':>16s}")
    for k in rows[0]:
        a = np.array([r[k][0] for r in rows]); e = np.array([r[k][1] for r in rows])
        print(f"  {k:10s} {a.mean():7.2f} +- {a.std():4.2f}   {e.mean():7.3f} +- {e.std():.3f}")

def paired(rows):
    d = np.array([r["DLA-NN"][0]-r["ILA-NN"][0] for r in rows])   # ACLR, paired by seed
    print(f"  paired DLA-ILA ACLR: mean {d.mean():+.2f} dB, sd(diff) {d.std(ddof=1):.2f}, "
          f"sign {int((d>0).sum())}/{len(d)} positive  (n={len(d)}, too few for significance)")

if __name__ == "__main__":
    import time; t0 = time.time()
    mp = [run(pa.MP_PA(), pa.torch_mp_pa(), s) for s in SEEDS]; summarize("MP PA (MATCHED)", mp); paired(mp)
    iq = [run(pa.IQMP_PA(), pa.torch_iqmp_pa(), s) for s in SEEDS]; summarize("MP + I/Q imbalance (MISMATCHED)", iq); paired(iq)

    # percent sweep, END-TO-END (the metric that matters)
    print("\n=== calibration sweep, end-to-end DLA-RVTDNN (MP PA) ===")
    import quant_export
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=11), 0.25); x_te = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
    PAnp = pa.MP_PA(); md = RVTDNN(4, 32, "relu"); train_dla(md, x_tr, pa.torch_mp_pa(), PAnp.c[0], epochs=EP, lr=1e-2)
    base = met(x_te, PAnp(nn_apply(md, x_te)))
    print(f"  float:  ACLR {base[0]:.2f}  EVM {base[1]:.4f}%")
    for pct in (100.0, 99.99, 99.9, 99.0):
        fwd, corr, info = quant_export.make_fixed_forward(md, x_tr, pct=pct)
        m = met(x_te, PAnp(fwd(x_te)))
        cfid = -dsp.nmse_db(info["y2"][:, 0]+1j*info["y2"][:, 1], corr(x_tr))
        print(f"  pct={pct:6.2f}: corr-fid {cfid:5.1f} dB | int16 ACLR {m[0]:.2f} ({m[0]-base[0]:+.2f}) EVM {m[1]:.4f}% ({m[1]-base[1]:+.4f})")
    print(f"\ntotal {time.time()-t0:.0f}s")

"""run.py - AI-DPD v2 experiment (addresses the algorithm/accuracy review).

  * ILA (indirect learning) vs DLA (direct learning through a differentiable PA)
  * multi-seed (5) -> mean +- std  (no single-run claims)
  * drive sweep (backoff curve)
  * end-to-end quantisation loss: float DPD vs int16 DPD (ACLR / EVM), which is
    the metric that actually matters - not the internal "SQNR".
"""
import numpy as np, torch, dsp, pa, quant_export
from models import RVTDNN
from train import train_nn, train_dla, nn_apply, poly_dpd_train, poly_dpd_apply

torch.manual_seed(0)
N, EPOCHS = 4096, 1500
SEEDS = [11, 22, 33, 44, 55]
TEST_SEED = 999

def met(x_te, y):
    lo, up = dsp.aclr_db(y)
    return min(lo, up), dsp.evm_pct(x_te, y)          # worst-side ACLR, EVM%

def one_run(PA_np, pa_t, seed):
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=seed), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=TEST_SEED), 0.25)
    G = PA_np.c[0]; y_tr = PA_np(x_tr)
    out = {}
    out["no-DPD"] = met(x_te, PA_np(x_te))
    c = poly_dpd_train(y_tr/G, x_tr)
    out["PolyMP"] = met(x_te, PA_np(poly_dpd_apply(x_te, c)))
    m_ila = RVTDNN(4, 32, "relu"); train_nn(m_ila, y_tr/G, x_tr, epochs=EPOCHS, lr=1e-2)
    out["ILA-RVTDNN"] = met(x_te, PA_np(nn_apply(m_ila, x_te)))
    m_dla = RVTDNN(4, 32, "relu"); train_dla(m_dla, x_tr, pa_t, G, epochs=EPOCHS, lr=1e-2)
    out["DLA-RVTDNN"] = met(x_te, PA_np(nn_apply(m_dla, x_te)))
    return out, m_dla

def summarize(name, per_seed):
    print(f"\n=== {name}: mean +- std over {len(SEEDS)} seeds ===")
    keys = list(per_seed[0].keys())
    print(f"  {'method':12s} {'ACLR(dB)':>16s} {'EVM(%)':>16s}")
    for k in keys:
        a = np.array([r[k][0] for r in per_seed]); e = np.array([r[k][1] for r in per_seed])
        print(f"  {k:12s} {a.mean():7.2f} +- {a.std():5.2f}   {e.mean():7.3f} +- {e.std():.3f}")

if __name__ == "__main__":
    import time
    t0 = time.time()
    # ---- scenario 1: MP PA ----
    mp_seeds = [one_run(pa.MP_PA(), pa.torch_mp_pa(), s)[0] for s in SEEDS]
    summarize("MP PA", mp_seeds)

    # ---- scenario 2: generalised MP PA ----
    gm_seeds = [one_run(pa.GmMP_PA(), pa.torch_gmmp_pa(), s)[0] for s in SEEDS]
    summarize("GmMP PA", gm_seeds)

    # ---- drive sweep (MP PA, one seed) ----
    print("\n=== drive sweep (MP PA, DLA vs Poly) ===")
    print(f"  {'drive':>6s} {'noDPD':>7s} {'Poly':>7s} {'ILA':>7s} {'DLA':>7s}   (ACLR dB)")
    for dr in (0.15, 0.20, 0.25, 0.30, 0.35):
        x_tr = dsp.scale(dsp.gen_ofdm(N, seed=11), dr)
        x_te = dsp.scale(dsp.gen_ofdm(N, seed=TEST_SEED), dr)
        PAnp = pa.MP_PA(); pat = pa.torch_mp_pa(); G = PAnp.c[0]
        c = poly_dpd_train(PAnp(x_tr)/G, x_tr)
        mi = RVTDNN(4, 32, "relu"); train_nn(mi, PAnp(x_tr)/G, x_tr, epochs=EPOCHS, lr=1e-2)
        md = RVTDNN(4, 32, "relu"); train_dla(md, x_tr, pat, G, epochs=EPOCHS, lr=1e-2)
        row = [met(x_te, PAnp(x_te))[0], met(x_te, PAnp(poly_dpd_apply(x_te, c)))[0],
               met(x_te, PAnp(nn_apply(mi, x_te)))[0], met(x_te, PAnp(nn_apply(md, x_te)))[0]]
        print(f"  {dr:6.2f} " + " ".join(f"{v:7.2f}" for v in row))

    # ---- end-to-end quantisation loss (DLA model, MP PA) ----
    print("\n=== quantisation loss (float DPD vs int16 DPD), DLA-RVTDNN ===")
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=11), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=TEST_SEED), 0.25)
    PAnp = pa.MP_PA(); pat = pa.torch_mp_pa(); G = PAnp.c[0]
    md = RVTDNN(4, 32, "relu"); train_dla(md, x_tr, pat, G, epochs=EPOCHS, lr=1e-2)
    torch.save(md.state_dict(), "RVTDNN_DLA.pt")
    int_fwd, int_corr, info = quant_export.make_fixed_forward(md, x_tr)
    a_float = met(x_te, PAnp(nn_apply(md, x_te)))
    a_int   = met(x_te, PAnp(int_fwd(x_te)))
    csnr = dsp.nmse_db(info["y2"][:, 0]+1j*info["y2"][:, 1], int_corr(x_tr))
    print(f"  float DPD : ACLR {a_float[0]:7.2f} dB  EVM {a_float[1]:.4f}%")
    print(f"  int16 DPD : ACLR {a_int[0]:7.2f} dB  EVM {a_int[1]:.4f}%")
    print(f"  loss      : {a_int[0]-a_float[0]:+.2f} dB ACLR   {a_int[1]-a_float[1]:+.4f} pp EVM")
    print(f"  correction fidelity (int16 vs float): {-csnr:.1f} dB")
    quant_export.quantize_and_export(md, x_tr)
    print(f"\ntotal {time.time()-t0:.0f}s")

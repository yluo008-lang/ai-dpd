"""run.py - full AI-DPD experiment: train, compare, quantize, export.

  scenarios : MP-PA (models match) and GmMP-PA (richer, non-trivial memory)
  models    : PolyMP (classical LS), RVTDNN, CVNN, TCN (neural)
  output    : comparison table + RVTDNN.pt + deploy/weights.h
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN, CVNN, TCN
from train import train_nn, nn_apply, poly_dpd_train, poly_dpd_apply
import quant_export

torch.manual_seed(0)

def evaluate(PA, x_tr, x_te, drive):
    y_tr, y_te = PA(x_tr), PA(x_te)
    G = PA.c[0]
    rows = {}
    rows["no-DPD"] = dsp.report("no-DPD", x_te, y_te)
    c = poly_dpd_train(y_tr/G, x_tr)
    rows["PolyMP"] = dsp.report("PolyMP-DPD", x_te, PA(poly_dpd_apply(x_te, c)))
    specs = [("RVTDNN", lambda: RVTDNN(4, 32, "relu"), 1e-2),
             ("CVNN",   lambda: CVNN(4, 24),            5e-3),
             ("TCN",    lambda: TCN(4, 16),             5e-3)]
    best = None
    for name, mk, lr in specs:
        m = train_nn(mk(), y_tr/G, x_tr, epochs=3000, lr=lr)
        z = nn_apply(m, x_te)
        rows[name] = dsp.report(name+"-DPD", x_te, PA(z))
        if name == "RVTDNN":
            torch.save(m.state_dict(), "RVTDNN.pt"); best = m
    return rows, best

if __name__ == "__main__":
    N, drive = 8192, 0.25
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=11), drive)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=33), drive)

    print("="*78); print("scenario 1: memory-polynomial PA (polynomial is the matched model)"); print("="*78)
    evaluate(pa.MP_PA(), x_tr, x_te, drive)

    print("\n"+"="*78); print("scenario 2: generalised MP PA (extra cross-memory, harder to invert)"); print("="*78)
    _, best = evaluate(pa.GmMP_PA(), x_tr, x_te, drive)

    print("\n"+"="*78); print("quantisation + deployment export (RVTDNN)"); print("="*78)
    quant_export.quantize_and_export(best, y_in=x_tr*0, x_cal=x_tr)

"""train.py - AI-DPD training via indirect learning (ILA), plus polynomial baseline.

  ILA:  train post-distorter  D(y/G) ~ x  ;  deploy as pre-distorter  z = D(x).
"""
import numpy as np, torch, torch.nn as nn
import dsp, pa
from models import RVTDNN, CVNN, TCN

torch.manual_seed(0)

# ------------------------------------------------------------------ polynomial DPD
def poly_dpd_train(y, x, M=4, Np=3, lam=1e-6):
    Phi = pa.mp_basis(y, M, Np)
    A = Phi.conj().T @ Phi + lam*np.trace(Phi.conj().T @ Phi)/Phi.shape[1]*np.eye(Phi.shape[1])
    b = Phi.conj().T @ x
    return np.linalg.solve(A, b)

def poly_dpd_apply(x, c, M=4, Np=3):
    return pa.mp_basis(x, M, Np) @ c

# ------------------------------------------------------------------ NN training
def train_nn(model, xin, target, epochs=200, lr=3e-3, log=None):
    # Train on the FULL contiguous sequence: delayed-tap features are causal
    # (zero-padded), so the fed samples must stay in true temporal order.
    xin_t = torch.as_tensor(xin, dtype=torch.complex64)
    tgt   = torch.as_tensor(target, dtype=torch.complex64)
    N = len(xin_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.mean(torch.abs(model(xin_t) - tgt)**2)
        loss.backward(); opt.step(); sch.step()
        if log and (ep % 40 == 0 or ep == epochs-1):
            print(f"    epoch {ep:3d}  train-MSE {loss.item():.3e}")
    return model

@torch.no_grad()
def nn_apply(model, x):
    model.eval()
    return model(torch.as_tensor(x, dtype=torch.complex64)).numpy().astype(complex)

# ------------------------------------------------------------------ direct learning (DLA)
def train_dla(model, x, pa_fn, gain, epochs=3000, lr=1e-2, log=None):
    """Direct learning: optimise || PA(D(x)) - G*x ||^2 with the PA in the loop.
    Eliminates the ILA domain mismatch (training input == deployment input)."""
    x_t = torch.as_tensor(x, dtype=torch.complex64)
    target = gain * x_t
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        y = pa_fn(model(x_t))
        loss = torch.mean(torch.abs(y - target)**2)
        loss.backward(); opt.step(); sch.step()
        if log and (ep % 200 == 0 or ep == epochs-1):
            print(f"    DLA epoch {ep:4d}  out-MSE {loss.item():.3e}")
    return model

# ------------------------------------------------------------------ full experiment
def run(seed_tr=11, seed_te=22, N=8192, drive=0.25, epochs=200):
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=seed_tr), drive)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=seed_te), drive)
    PA = pa.MP_PA()
    y_tr = PA(x_tr); y_te = PA(x_te)
    G = PA.c[0]
    yn_tr = y_tr / G

    print("\n[baseline] no DPD (test set):"); base = dsp.report("no-DPD", x_te, y_te)
    print("\n[poly] memory-polynomial LS DPD")
    c = poly_dpd_train(yn_tr, x_tr)
    z_te = poly_dpd_apply(x_te, c); dsp.report("PolyMP-DPD", x_te, PA(z_te))

    results = {"no-DPD": base}
    models = {"RVTDNN": RVTDNN(4, 32, "tanh"),
              "CVNN":   CVNN(4, 24),
              "TCN":    TCN(4, 16)}
    for name, m in models.items():
        print(f"\n[train] {name}")
        train_nn(m, yn_tr, x_tr, epochs=epochs)
        z = nn_apply(m, x_te)
        results[name] = dsp.report(name+"-DPD", x_te, PA(z))
        torch.save(m.state_dict(), f"{name}.pt")
    return results, c

if __name__ == "__main__":
    run()

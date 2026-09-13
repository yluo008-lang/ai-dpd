"""quant_tcn_export.py - int16 quantisation of the TCN DPD, per-channel.

Per-tensor activation scaling destroys the correction (the conv output channels
have very different dynamic ranges). This version uses PER-OUTPUT-CHANNEL weight
and activation scales:
    real[o] = sw[o] * sum_c sx[c]*(sum_j Wq[o,c,j]*xq[c,idx]) + b[o]
    a[o]    = round(relu(real[o]) / sy[o])
Also exports the C header + reference I/O, and prints per-tensor vs per-channel
correction fidelity / end-to-end loss.
"""
import numpy as np, torch, dsp, pa
from models import TCN

def q(a, s):  return np.clip(np.round(a/s), -32768, 32767).astype(np.int64)

def conv_fixed(xq, sx, Wq, sw, b, sy, d, relu):
    Cout, Cin, K = Wq.shape; N = xq.shape[1]
    out = np.zeros((Cout, N), dtype=np.int64)
    for o in range(Cout):
        acc = np.zeros(N)
        for c in range(Cin):
            s = np.zeros(N, dtype=np.int64)
            for j in range(K):
                sh = d*(2-j)
                xs = np.concatenate([np.zeros(sh, np.int64), xq[c][:N-sh]]) if sh > 0 else xq[c]
                s = s + Wq[o, c, j]*xs
            acc = acc + sx[c]*s
        real = sw[o]*acc + b[o]
        if relu: real = np.maximum(real, 0.0)
        out[o] = np.clip(np.round(real/sy[o]), -32768, 32767)
    return out

def mode_scales(x, mode):
    return mode            # "per_tensor" or "per_channel"

def run_fixed(model, x, per_channel=True):
    sd = model.state_dict()
    nL   = len(model.convs)
    dils = [int(m.dilation[0]) for m in model.convs]
    W  = [sd[f"convs.{i}.weight"].numpy().astype(np.float64) for i in range(nL)]
    B  = [sd[f"convs.{i}.bias"].numpy().astype(np.float64) for i in range(nL)]
    Wh = sd["head.weight"].numpy().astype(np.float64); Bh = sd["head.bias"].numpy().astype(np.float64)
    N = len(x)
    xi = np.round(x.real*32768).astype(np.int64); xq_ = np.round(x.imag*32768).astype(np.int64)
    xr = np.stack([xi, xq_])/32768.0
    x0 = np.stack([xi, xq_])
    # input scales
    if per_channel:
        SX0 = np.abs(xr).max(axis=1)/32700.0                       # (2,)
    else:
        SX0 = np.full(2, np.abs(xr).max()/32700.0)
    a = np.stack([q(xr[0], SX0[0]), q(xr[1], SX0[1])])
    layers = []
    sx = SX0
    for i in range(nL):
        d = dils[i]
        if per_channel:
            sw = np.abs(W[i].reshape(W[i].shape[0], -1)).max(axis=1)/32767.0   # (Cout,)
        else:
            sw = np.full(W[i].shape[0], np.abs(W[i]).max()/32767.0)
        Wq = np.stack([q(W[i][o], sw[o]) for o in range(W[i].shape[0])])
        # float activation to determine sy
        Cout, Cin, K = W[i].shape
        real = np.zeros((Cout, N))
        for o in range(Cout):
            acc = np.zeros(N)
            for c in range(Cin):
                s = np.zeros(N)
                for j in range(K):
                    sh = d*(2-j)
                    xs = np.concatenate([np.zeros(sh), (a[c]*sx[c])[:N-sh]]) if sh > 0 else a[c]*sx[c]
                    s = s + Wq[o, c, j]*sw[o]*xs
                acc = acc + s
            real[o] = acc + B[i][o]
        # sy from PRE-relu magnitude (a channel that is all-negative would give sy=0)
        sy = (np.abs(real).max(axis=1)/32700.0) if per_channel else np.full(Cout, np.abs(real).max()/32700.0)
        sy = np.maximum(sy, 1e-12)
        a = conv_fixed(a, sx, Wq, sw, B[i], sy, d, True)
        layers.append(dict(d=d, Wq=Wq, sw=sw, sy=sy, b=B[i]))
        sx = sy
    # head
    SWH = np.abs(Wh).max()/32767.0
    Whq = q(Wh, SWH)
    hr = np.zeros((2, N)); hi = np.zeros((2, N))
    real_h = np.zeros((2, N))
    for o in range(2):
        for c in range(32):
            real_h[o] += (Whq[o, c]*SWH)*(a[c]*sy[c])
        real_h[o] += Bh[o]
    corr = real_h[0] + 1j*real_h[1]
    out = corr + (xr[0] + 1j*xr[1])
    return out, corr, dict(layers=layers, SX0=SX0, SWH=SWH, Whq=Whq, Bh=Bh, x0=x0, N=N)

def fidelity(model, x):
    with torch.no_grad():
        zf = model(torch.as_tensor(x, dtype=torch.complex64)).numpy().astype(complex)
    return zf

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

if __name__ == "__main__":
    m = TCN(4, 32, (1, 2, 4, 8)); m.load_state_dict(torch.load("TCN_DLA.pt"))
    x = dsp.scale(dsp.gen_ofdm(1024, seed=11), 0.25)
    PA = pa.MP_PA()
    zf = fidelity(m, x)
    print(f"float TCN : ACLR {met(x, PA(zf))[0]:7.2f}  EVM {met(x, PA(zf))[1]:.4f}%")
    for pc in (False, True):
        out, corr, info = run_fixed(m, x, per_channel=pc)
        cf = dsp.nmse_db(zf - x, (out - x))
        print(f"    rms corr_float={np.sqrt(np.mean(np.abs(zf-x)**2)):.5f} "
              f"corr_fixed={np.sqrt(np.mean(np.abs(out-x)**2)):.5f} "
              f"diff={np.sqrt(np.mean(np.abs((zf-x)-(out-x))**2)):.5f}")
        a = met(x, PA(out))
        print(f"{'per-channel' if pc else 'per-tensor ':10s}: correction fid {-cf:5.1f} dB | "
              f"int16 ACLR {a[0]:7.2f} ({a[0]-met(x,PA(zf))[0]:+.2f})  EVM {a[1]:.4f}%")

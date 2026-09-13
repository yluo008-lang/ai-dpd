"""quant_tcn.py - int16 quantisation of the TCN DPD + end-to-end loss.

The TCN is the strongest model (ACLR -54.8). Here we quantise its weights (conv +
linear) to int16 per-tensor and measure the END-TO-END ACLR/EVM loss -- the metric
that matters -- not an internal SQNR.
"""
import numpy as np, torch, torch.nn as nn, dsp, pa
from models import TCN
from train import train_dla, nn_apply

torch.manual_seed(0)
N, EP, TS = 8192, 3000, 999

def dequant_state(model, bits=16):
    qmax = 2**(bits-1) - 1
    sd = model.state_dict()
    out, scales = {}, {}
    for k, v in sd.items():
        if v.dtype.is_floating_point and v.numel() > 1 and "weight" in k or ("weight" in k or "bias" in k):
            pass
    for k, v in sd.items():
        if "weight" in k:                       # quantise weights per-tensor
            s = float(v.abs().max()) / qmax
            q = torch.clamp(torch.round(v / s), -qmax-1, qmax)
            out[k] = (q * s).to(v.dtype); scales[k] = s if s > 0 else 1.0
        else:
            out[k] = v                          # keep biases float
    return out, scales

def quant_forward(model, sd_q, x):
    bak = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(sd_q, strict=True)
    with torch.no_grad():
        y = model(torch.as_tensor(x, dtype=torch.complex64)).numpy().astype(complex)
    model.load_state_dict(bak, strict=True)
    return y

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

if __name__ == "__main__":
    PA = pa.MP_PA(); G = PA.c[0]
    x_tr = dsp.scale(dsp.gen_ofdm(N, seed=11), 0.25)
    x_te = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
    m = TCN(4, 32, (1, 2, 4, 8))
    train_dla(m, x_tr, pa.torch_mp_pa(), G, epochs=EP, lr=1e-3)
    torch.save(m.state_dict(), "TCN_DLA.pt")

    sd_f = m.state_dict()
    sd_q, scales = dequant_state(m, 16)
    zf = nn_apply(m, x_te); zq = quant_forward(m, sd_q, x_te)
    yf, yq = PA(zf), PA(zq)
    af, aq = met(x_te, yf), met(x_te, yq)
    cfid = dsp.nmse_db(np.mean([v for v in [0]]), [0])  # placeholder unused
    # correction fidelity (float vs int16) on the correction only
    corr = dsp.nmse_db(np.ones(1), np.ones(1)) if False else None
    import numpy as np2
    cf = np.mean(np.abs((zf - x_te) - (zq - x_te))**2)
    pw = np.mean(np.abs(zf - x_te)**2)
    print("=== TCN-32x(1,2,4,8) int16 weight quantisation (MP PA) ===")
    print(f"  float DPD : ACLR {af[0]:7.2f} dB  EVM {af[1]:.4f}%")
    print(f"  int16 DPD : ACLR {aq[0]:7.2f} dB  EVM {aq[1]:.4f}%")
    print(f"  loss      : {aq[0]-af[0]:+.2f} dB ACLR   {aq[1]-af[1]:+.4f} pp EVM")
    print(f"  correction fidel (weight-only int16): {-10*np.log10(cf/pw):.1f} dB")
    print(f"  #quantised tensors: {len(scales)}")

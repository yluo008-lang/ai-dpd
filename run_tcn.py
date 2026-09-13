"""run_tcn.py - diagnose why TCN ACLR sits at the no-DPD floor while its EVM is good.

Hypothesis: EVM is in-band and ACLR is adjacent-band. A conv-based correction can
improve in-band while splattering out-of-band, so ACLR stays at the floor.
Measure the DPD correction's in-band vs out-of-band power.
"""
import numpy as np, torch, dsp, pa
from models import RVTDNN, TCN
from train import train_nn, nn_apply

torch.manual_seed(0)
N, EP, TS = 8192, 3000, 999

def band_split(sig, os=4):
    Y = np.fft.fft(sig); P = np.abs(Y)**2; Nl = len(Y)
    ch = Nl//os; lo = Nl//2 - ch//2; hi = lo + ch
    inb = P[lo:hi].sum(); return inb, P.sum()-inb

def met(x, y):
    lo, up = dsp.aclr_db(y); return min(lo, up), dsp.evm_pct(x, y)

if __name__ == "__main__":
    xtr = dsp.scale(dsp.gen_ofdm(N, seed=11), 0.25)
    xte = dsp.scale(dsp.gen_ofdm(N, seed=TS), 0.25)
    PA = pa.MP_PA(); G = PA.c[0]
    print(f"no-DPD: ACLR {met(xte,PA(xte))[0]:.2f}  EVM {met(xte,PA(xte))[1]:.3f}%")
    for name, mk, lr in [("RVTDNN",   lambda: RVTDNN(4,32,"relu"), 1e-2),
                         ("TCN-16",   lambda: TCN(4,16),           3e-3),
                         ("TCN-32x4", lambda: TCN(4,32,(1,2,4,8)), 1e-3)]:
        m = train_nn(mk(), PA(xtr)/G, xtr, epochs=EP, lr=lr)
        z = nn_apply(m, xte); y = PA(z); corr = z - xte
        acl, evm = met(xte, y)
        ci, co = band_split(corr)
        print(f"{name:9s}: ACLR {acl:6.2f}  EVM {evm:6.3f}%  | correction rms {np.sqrt(np.mean(np.abs(corr)**2)):.4f} "
              f"oob/inb {co/ci:.2e}")

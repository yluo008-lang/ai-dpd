"""gen_feats.py - golden vectors for the RTL feature builder (nn_dpd_feat.v).

Writes deploy/iq.hex (input stream: I Q as 4-hex) and deploy/feat.hex
(expected 12 features per sample: I_m,Q_m,|x_m|), matching models.real_features.
"""
import numpy as np, dsp, torch
from models import real_features

N, M = 64, 4
x = dsp.scale(dsp.gen_ofdm(N, seed=5), 0.25)
q = lambda v: int(np.clip(round(v*32768), -32768, 32767))
xi = np.array([q(v.real) for v in x]); xq = np.array([q(v.imag) for v in x])
# expected features computed exactly as the RTL does: I,Q,floor(sqrt(I^2+Q^2)) on int16
isqrt = lambda v: int(np.floor(np.sqrt(max(v, 0))))
feat = []
for n in range(N):
    row = []
    for m in range(M):
        k = n-m
        I = xi[k] if k >= 0 else 0
        Q = xq[k] if k >= 0 else 0
        row += [I, Q, isqrt(I*I+Q*Q)]
    feat.append(row)
feat = np.array(feat)

with open("deploy/iq.hex", "w") as f:
    for i in range(N):
        f.write(f"{xi[i] & 0xFFFF:04x} {xq[i] & 0xFFFF:04x}\n")
with open("deploy/feat.hex", "w") as f:
    for i in range(N):
        f.write(" ".join(f"{int(v) & 0xFFFF:04x}" for v in feat[i]) + "\n")
print(f"wrote deploy/iq.hex + feat.hex ({N} samples, {M} taps)")

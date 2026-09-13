"""gen_par_ref.py - golden for the FULL parallel DPD chain (feat->scale->L0->L1->L2+residual).

Emits par_params.vh (K as Q13) and par_ref.hex ({I_out,Q_out} per sample, Q1.15).
Everything is integer arithmetic identical to the RTL.
"""
import re, numpy as np

KSH = 13
def arr(h, name):
    m = re.search(name + r"\[[^\]]*\]\s*=\s*\{([^}]*)\}", h, re.S)
    return np.array([int(x) for x in re.findall(r"-?\d+", m.group(1))])
def s16(v):  return v - 65536 if v >= 32768 else v
def lyr(a, W, BQ, REQ, SH, relu=True):
    t = (a @ W.T)*REQ + (BQ << SH) + (1 << (SH-1))
    v = t >> SH
    if relu: v = np.maximum(v, 0)
    return np.clip(v, -32768, 32767)

def main():
    h = open("deploy/weights.h").read()
    W0 = arr(h, "W0q").reshape(32, 12); W1 = arr(h, "W1q").reshape(32, 32); W2 = arr(h, "W2q").reshape(2, 32)
    BQ0 = arr(h, "BQ0q"); BQ1 = arr(h, "BQ1q"); BQ2 = arr(h, "BQ2q")
    REQ0 = int(re.search(r"#define REQ0 (\d+)", h).group(1))
    REQ1 = int(re.search(r"#define REQ1 (\d+)", h).group(1))
    REQ2 = int(re.search(r"#define REQ2 (\d+)", h).group(1))
    SH   = int(re.search(r"#define SHIFTQ (\d+)", h).group(1))
    M    = int(re.search(r"#define NNE_M (\d+)", h).group(1))
    isqrt = lambda v: int(np.floor(np.sqrt(max(v, 0))))

    iq = [l.split() for l in open("deploy/iq.hex").read().split("\n") if l.strip()]
    xi = [s16(int(a, 16)) for a, b in iq]; xq = [s16(int(b, 16)) for a, b in iq]
    N = len(xi)
    feats = []
    for n in range(N):
        row = []
        for m in range(M):
            k = n - m; I = xi[k] if k >= 0 else 0; Q = xq[k] if k >= 0 else 0
            row += [I, Q, isqrt(I*I + Q*Q)]
        feats.append(row)
    feats = np.array(feats, dtype=np.int64)
    K = 32700.0/max(1, np.abs(feats).max()); KQ = int(round(K*(1 << KSH)))

    a0 = np.clip((feats*KQ) >> KSH, -32768, 32767)
    a1 = lyr(a0, W0, BQ0, REQ0, SH)
    a2 = lyr(a1, W1, BQ1, REQ1, SH)
    c  = lyr(a2, W2, BQ2, REQ2, SH, relu=False)                     # (N,2) Q1.15 correction
    out = np.clip(c + np.stack([np.array(xi), np.array(xq)], axis=1), -32768, 32767)

    with open("deploy/par_ref.hex", "w") as f:
        for n in range(N):
            f.write(f"{int(out[n,0]) & 0xFFFF:04x} {int(out[n,1]) & 0xFFFF:04x}\n")
    with open("deploy/par_params.vh", "w") as f:
        f.write(f"`define PAR_KQ {KQ}\n`define PAR_KSH {KSH}\n")
    print(f"KQ={KQ}  wrote par_ref.hex ({N} lines) + par_params.vh")

if __name__ == "__main__":
    main()

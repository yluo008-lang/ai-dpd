"""gen_par_ref.py - golden + params for the parallel DPD chain (feat -> scale -> L0).

The TCN/MLP were trained with activations a0 = round(feat_real / SX0). The RTL
feature builder emits feat in Q1.15, so the MLP needs a scaling stage:
    a0 = round(feat_q15 * K),   K = 32700 / max|feat_q15|
Emit par_params.vh (K as Q13) and par_ref.hex (expected layer-0 output per sample).
"""
import re, numpy as np

KSH = 13
def arr(h, name):
    m = re.search(name + r"\[[^\]]*\]\s*=\s*\{([^}]*)\}", h, re.S)
    return np.array([int(x) for x in re.findall(r"-?\d+", m.group(1))])

def s16(v):  return v-65536 if v >= 32768 else v

def main():
    h = open("deploy/weights.h").read()
    W0 = arr(h, "W0q").reshape(32, 12); BQ0 = arr(h, "BQ0q")
    REQ0 = int(re.search(r"#define REQ0 (\d+)", h).group(1))
    SH = int(re.search(r"#define SHIFTQ (\d+)", h).group(1))
    M = int(re.search(r"#define NNE_M (\d+)", h).group(1))
    isqrt = lambda v: int(np.floor(np.sqrt(max(v, 0))))

    iq = [l.split() for l in open("deploy/iq.hex").read().split("\n") if l.strip()]
    xi = [s16(int(a, 16)) for a, b in iq]; xq = [s16(int(b, 16)) for a, b in iq]
    N = len(xi)
    feats = []
    for n in range(N):
        row = []
        for m in range(M):
            k = n - m
            I = xi[k] if k >= 0 else 0; Q = xq[k] if k >= 0 else 0
            row += [I, Q, isqrt(I*I + Q*Q)]
        feats.append(row)
    feats = np.array(feats, dtype=np.int64)
    K = 32700.0/max(1, np.abs(feats).max())
    KQ = int(round(K * (1 << KSH)))
    print(f"max|feat|={np.abs(feats).max()}  K={K:.4f}  KQ={KQ} (Q{KSH})")

    a0 = np.clip((feats*KQ) >> KSH, -32768, 32767).astype(np.int64)   # exact RTL arithmetic
    print("DBG py feats[0][0:3]=", feats[0][0:3], " a0[0:3]=", a0[0][0:3])
    acc = a0 @ W0.T
    t = acc*REQ0 + (BQ0 << SH) + (1 << (SH-1))
    v = t >> SH; v = np.maximum(v, 0); v = np.clip(v, -32768, 32767)
    with open("deploy/par_ref.hex", "w") as f:
        for n in range(N):
            f.write(" ".join(f"{int(x) & 0xFFFF:04x}" for x in v[n]) + "\n")
    with open("deploy/par_params.vh", "w") as f:
        f.write(f"`define PAR_KQ {KQ}\n`define PAR_KSH {KSH}\n")
    print(f"wrote deploy/par_ref.hex ({N} lines) + par_params.vh")

if __name__ == "__main__":
    main()

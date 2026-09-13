"""quant_tcn_export.py - export the trained TCN as int16 weights + a reference I/O
for the C and RTL cross-checks (conv1d + ReLU + linear + residual).

int16 scheme: per-tensor weight scale (max/32767) and per-tensor activation scale
(max/32700); integer accumulate then requant. Mirrors deploy/tcn_infer.c.
"""
import numpy as np, torch, dsp, pa
from models import TCN

SHIFT = 30

def q(a, s):  return np.clip(np.round(a/s), -32768, 32767).astype(np.int64)

def conv_fixed(xq, Wq, b, sw, sx, sy, d, relu):
    """xq: (Cin,N) int; Wq: (Cout,Cin,3) int; returns (Cout,N) int16."""
    Cout, Cin, K = Wq.shape; N = xq.shape[1]
    out = np.zeros((Cout, N), dtype=np.int64)
    for o in range(Cout):
        acc = np.zeros(N, dtype=np.int64)
        for c in range(Cin):
            for j in range(K):
                sh = d*(2-j)                       # out[t] uses x[t + d*j - 2d]
                xc = xq[c]
                sh_x = np.concatenate([np.zeros(sh, dtype=np.int64), xc[:N-sh]]) if sh > 0 else xc
                acc += Wq[o, c, j] * sh_x
        # dequantise to real, add bias, requantise to sy
        real = acc*(sw*sx) + b[o]
        if relu: real = np.maximum(real, 0.0)
        out[o] = np.clip(np.round(real/sy), -32768, 32767)
    return out

def export(model, x, out_c="deploy/tcn_weights.h", out_io="deploy/tcn_ref_io.txt"):
    import os
    sd = model.state_dict()
    W = [sd[f"convs.{i}.weight"].numpy().astype(np.float64) for i in range(3)]
    B = [sd[f"convs.{i}.bias"].numpy().astype(np.float64) for i in range(3)]
    Wh = sd["head.weight"].numpy().astype(np.float64); Bh = sd["head.bias"].numpy().astype(np.float64)

    # weights -> int16
    SW = [float(np.abs(w).max())/32767.0 for w in W]
    Whs = float(np.abs(Wh).max())/32767.0
    Wq = [q(w, s) for w, s in zip(W, SW)]
    Whq = q(Wh, Whs)

    N = len(x)
    xi = np.round(x.real*32768).astype(np.int64); xq_ = np.round(x.imag*32768).astype(np.int64)
    x0 = np.stack([xi, xq_])                                   # (2,N) int (Q15)
    xr = np.stack([x.real, x.imag])                            # (2,N) real units
    SX0 = float(np.abs(xr).max())/32700.0
    a0 = q(xr, SX0)
    # conv0
    SX1 = 1.0/ (32700.0/ np.abs(np.stack([xi,xq_])).max()) if False else 0
    # compute activations float to pick scales (use a running max)
    convs = []
    a = a0
    sx_prev = SX0
    layer_out = []
    for i in range(3):
        d = [1, 2, 4][i]
        # scale for this layer's output: estimate from float activations
        real = None
        out_int = conv_fixed(a, Wq[i], B[i], SW[i], sx_prev, 1.0/32700.0, d, True)  # provisional sy
        # recompute with proper sy = max|real|/32700
        # (do one pass in real units)
        Cout, Cin, K = Wq[i].shape
        outr = np.zeros((Cout, N))
        for o in range(Cout):
            for c in range(Cin):
                for j in range(K):
                    sh = d*(2-j)
                    xc = a[c]
                    sh_x = np.concatenate([np.zeros(sh, dtype=np.int64), xc[:N-sh]]) if sh > 0 else xc
                    outr[o] += (Wq[i][o, c, j]*SW[i])*(sh_x*sx_prev)
            outr[o] += B[i][o]
        outr = np.maximum(outr, 0)
        sy = float(outr.max())/32700.0
        a = conv_fixed(a, Wq[i], B[i], SW[i], sx_prev, sy, d, True)
        layer_out.append(a.copy())
        convs.append(dict(d=d, sy=sy, sw=SW[i]))
        sx_prev = sy
    # head: Linear(32->2), no relu; residual added in real units
    Sh = convs[2]['sy']                                   # scale of last conv output
    hq = a
    real_h = np.zeros((2, N))
    for o in range(2):
        for c in range(32):
            real_h[o] += (Whq[o, c]*Whs)*(hq[c]*Sh)
        real_h[o] += Bh[o]
    corr = real_h[0] + 1j*real_h[1]
    out = corr + x

    os.makedirs(os.path.dirname(out_c), exist_ok=True)
    with open(out_c, "w") as f:
        f.write("// auto-generated TCN int16 weights\n#pragma once\n")
        def arr(name, A, t="short"):
            f.write(f"static const {t} {name}[{A.size}] = {{\n")
            fl = A.reshape(-1)
            for i in range(0, fl.size, 12):
                f.write("  " + ",".join(str(int(v)) for v in fl[i:i+12]) + ",\n")
            f.write("};\n")
        for i in range(3):
            arr(f"TCN_W{i}", Wq[i])                        # shape (Cout,Cin,K), row-major
            arr(f"TCN_B{i}", B[i], "double")
            f.write(f"static const double TCN_SW{i} = {SW[i]!r};\n")
            f.write(f"static const double TCN_SY{i} = {convs[i]['sy']!r};\n")
            f.write(f"#define TCN_D{i} {convs[i]['d']}\n")
        arr("TCN_WH", Whq); arr("TCN_BH", Bh, "double")
        f.write(f"static const double TCN_SWH = {Whs!r};\nstatic const double TCN_SH = {Sh!r};\n")
        f.write(f"static const double TCN_SX0 = {SX0!r};\n#define TCN_SHIFT {SHIFT}\n#define TCN_N {N}\n")
    with open(out_io, "w") as f:
        for t in range(min(N, 32)):
            f.write(f"{xi[t]&0xFFFF:04x} {xq_[t]&0xFFFF:04x} {int(np.round(out[t].real*32768))&0xFFFF:04x} {int(np.round(out[t].imag*32768))&0xFFFF:04x}\n")
    with open("deploy/tcn_dbg.txt", "w") as f:
        for i in range(3):
            f.write(" ".join(str(int(v)) for v in layer_out[i][:4, 0]) + "\n")   # layer i, ch0..3, t=0
    print(f"exported TCN -> {out_c}, {out_io}  (N={N})")

if __name__ == "__main__":
    m = TCN(4, 32, (1, 2, 4, 8))
    m.load_state_dict(torch.load("TCN_DLA.pt"))
    x = dsp.scale(dsp.gen_ofdm(256, seed=11), 0.25)
    export(m, x)

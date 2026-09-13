"""models.py - neural network DPD architectures (PyTorch).

Three representative families from the recent literature:
  RVTDNN : real-valued time-delay neural network (MLP over I/Q/envelope taps)
  CVNN   : complex-valued NN (complex linear layers + modReLU)
  TCN    : causal temporal convolutional net over the complex baseband stream
All are causal with memory depth M, so they map 1:1 onto a streaming datapath.
"""
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------- feature build
def real_features(x, M):
    """x: complex np/torch (N,) -> real tensor (N, 3M): [I_m, Q_m, |x_m|] per tap."""
    if torch.is_tensor(x):
        xr, xi = x.real, x.imag
    else:
        x = torch.as_tensor(x); xr, xi = x.real, x.imag
    N = x.shape[0]
    feats = []
    for m in range(M):
        r = torch.roll(xr, m); i = torch.roll(xi, m)      # roll = delay (wrap is fine, seq is long)
        feats += [r, i, torch.sqrt(r*r + i*i + 1e-12)]
    return torch.stack(feats, dim=1)                       # (N, 3M)

def complex_taps(x, M):
    """x: complex torch (N,) -> complex tensor (N, M) of delayed samples."""
    cols = []
    for m in range(M):
        cols.append(torch.roll(x, m))
    return torch.stack(cols, dim=1)

# ---------------------------------------------------------------- models
class RVTDNN(nn.Module):
    def __init__(self, M=4, H=32, act="tanh", residual=True):
        super().__init__()
        a = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}[act]
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(3*M, H), a,
            nn.Linear(H, H),   a,
            nn.Linear(H, 2),
        )
    def forward(self, x):
        out = self.net(real_features(x, self.M))           # (N,2)
        y = torch.complex(out[:, 0], out[:, 1])
        return y + x if self.residual else y
    @property
    def M(self): return self.net[0].in_features // 3

class modReLU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(n))
    def forward(self, z):
        mag = torch.abs(z) + 1e-8
        return torch.relu(mag + self.b) * (z / mag)

class CLinear(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.wr = nn.Linear(nin, nout); self.wi = nn.Linear(nin, nout)
    def forward(self, z):
        return self.wr(z.real) - self.wi(z.imag) + 1j*(self.wr(z.imag) + self.wi(z.real))

class CVNN(nn.Module):
    def __init__(self, M=4, H=24, residual=True):
        super().__init__()
        self.m = M
        self.residual = residual
        self.c1 = CLinear(M, H);  self.a1 = modReLU(H)
        self.c2 = CLinear(H, H);  self.a2 = modReLU(H)
        self.c3 = CLinear(H, 1)
    def forward(self, x):
        z = complex_taps(x, self.m)                        # (N,M) complex
        z = self.a1(self.c1(z))
        z = self.a2(self.c2(z))
        out = self.c3(z)[:, 0]
        return out + x if self.residual else out
    @property
    def M(self): return self.m

class TCN(nn.Module):
    """Causal dilated conv (kernel 3, dilations 1,2,4) over (I,Q) + final MLP."""
    seq_model = True
    def __init__(self, M=4, C=16, layers=(1,2,4), residual=True):
        super().__init__()
        self.m = M
        self.residual = residual
        self.convs = nn.ModuleList()
        cin = 2
        for d in layers:
            self.convs.append(nn.Conv1d(cin, C, 3, dilation=d, padding=0))
            cin = C
        self.act = nn.ReLU()
        self.head = nn.Linear(C, 2)
        self.pad = 2*sum(layers)                                # 2*sum = causal left pad
    def forward(self, x):
        seq = torch.stack([x.real, x.imag], dim=0).unsqueeze(0)   # (1,2,N)
        seq = nn.functional.pad(seq, (self.pad, 0))
        for c in self.convs:
            seq = self.act(c(seq))
        seq = seq.squeeze(0).transpose(0, 1)                    # (N,C)
        out = self.head(seq)
        y = torch.complex(out[:, 0], out[:, 1])
        return y + x if self.residual else y                    # residual: lower bound == no-DPD
        return torch.complex(out[:, 0], out[:, 1])
    @property
    def M(self): return self.m

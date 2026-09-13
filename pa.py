"""pa.py - behavioural PA models for DPD development.

  MP_PA   : memory polynomial (deterministic, fast, differentiable)
  GmMP_PA : generalised memory polynomial (adds lagging/leading cross terms)
  plus optional thermal/DC memory effect and AWGN.
"""
import numpy as np

PA_COEF = np.array([
    # m=0
    1.000+0.000j, -0.130+0.060j, -0.030-0.020j,
    # m=1
    0.000+0.030j, -0.020-0.010j,  0.010+0.005j,
    # m=2
    0.000-0.020j,  0.010+0.000j,  0.000+0.010j,
    # m=3
    0.000+0.010j,  0.000-0.005j,  0.000+0.000j,
], dtype=complex)

def mp_basis(x, M=4, Np=3):
    """Memory-polynomial basis, column k = m*Np + j  (order k = 2j+1)."""
    N = len(x)
    Phi = np.zeros((N, M*Np), dtype=complex)
    for m in range(M):
        xm = np.concatenate([np.zeros(m, complex), x[:N-m]]) if m else x
        mag2 = np.abs(xm)**2
        t = xm.copy()
        for j in range(Np):
            Phi[:, m*Np+j] = t
            t = t * mag2
    return Phi

class MP_PA:
    def __init__(self, coef=PA_COEF, M=4, Np=3, noise=0.0):
        self.c, self.M, self.Np, self.noise = coef, M, Np, noise
    def __call__(self, x, rng=None):
        y = mp_basis(x, self.M, self.Np) @ self.c
        if self.noise > 0:
            rng = rng or np.random.default_rng(0)
            n = (rng.standard_normal(len(x)) + 1j*rng.standard_normal(len(x)))
            y = y + self.noise*n*np.sqrt(np.mean(np.abs(y)**2))
        return y

class GmMP_PA:
    """Generalised MP: adds cross terms x(n-m)*|x(n-l)|^2 (l != m)."""
    def __init__(self, M=4, Np=3, seed=7):
        rng = np.random.default_rng(seed)
        self.M, self.Np = M, Np
        self.c = np.zeros(M*Np, complex)
        self.c[0] = 1.0
        self.c[1] = -0.11+0.05j
        self.c[2] = -0.02-0.015j
        for m in range(1, M):
            self.c[m*Np]   = 0.02*(rng.standard_normal()+1j*rng.standard_normal())
            self.c[m*Np+1] = 0.01*(rng.standard_normal()+1j*rng.standard_normal())
        # cross terms: for tap m, envelope from previous tap
        self.cc = 0.03*(rng.standard_normal(M)+1j*rng.standard_normal(M))
    def __call__(self, x):
        y = mp_basis(x, self.M, self.Np) @ self.c
        N = len(x)
        for m in range(1, self.M):
            xm = np.concatenate([np.zeros(m, complex), x[:N-m]])
            xp = np.concatenate([np.zeros(m-1, complex), x[:N-(m-1)]]) if m >= 1 else x
            y += self.cc[m] * xm * np.abs(xp)**2
        return y

if __name__ == "__main__":
    import dsp
    x = dsp.scale(dsp.gen_ofdm(seed=3), 0.25)
    y = MP_PA()(x)
    dsp.report("PA(no DPD)", x, y)

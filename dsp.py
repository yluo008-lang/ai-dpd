"""dsp.py - OFDM test signal + PA behavioural metrics (ACLR / EVM / NMSE).

Pure NumPy; used by the training and evaluation code.
"""
import numpy as np

def gen_ofdm(N=8192, os=4, seed=1, qam=16):
    """16-QAM OFDM occupying the central 1/os of the Nyquist band (os x oversampled)."""
    rng = np.random.default_rng(seed)
    X = np.zeros(N, dtype=complex)
    ch = N // os
    lo = N // 2 - ch // 2
    k = int(np.sqrt(qam))
    re = rng.integers(0, k, ch)
    im = rng.integers(0, k, ch)
    X[lo:lo+ch] = (2*re - (k-1)) + 1j*(2*im - (k-1))
    x = np.fft.ifft(X)
    x /= np.sqrt(np.mean(np.abs(x)**2))
    return x

def scale(x, rms): return x * (rms / np.sqrt(np.mean(np.abs(x)**2)))

def nmse_db(ref, meas):
    """LS-gain-normalised NMSE in dB."""
    g = np.vdot(ref, meas) / np.vdot(ref, ref)      # optimal complex gain
    e = meas - g*ref
    return 10*np.log10(np.sum(np.abs(e)**2) / np.sum(np.abs(g*ref)**2))

def evm_pct(ref, meas, os=4):
    """EVM %, band-limited to the signal channel, LS-fit gain."""
    N = len(meas)
    Y = np.fft.fft(meas)
    ch = N // os; lo = N//2 - ch//2; hi = lo + ch
    m = np.zeros(N, dtype=complex)
    m[lo:hi] = Y[lo:hi]
    mo = np.fft.ifft(m)
    g = np.vdot(ref, mo) / np.vdot(ref, ref)
    e = mo - g*ref
    return 100*np.sqrt(np.sum(np.abs(e)**2) / np.sum(np.abs(g*ref)**2))

def aclr_db(y, os=4):
    """Return (ACLR_low, ACLR_up) in dB (Padj/Pch)."""
    N = len(y)
    Y = np.fft.fft(y)
    P = np.abs(Y)**2
    ch = N // os; c = N//2
    mlo = c - ch//2; mhi = mlo + ch
    alo, ahi = mlo - ch, mlo
    blo, bhi = mhi, mhi + ch
    pch = P[mlo:mhi].sum()
    plo = P[alo:ahi].sum()
    pup = P[blo:bhi].sum()
    return 10*np.log10(plo/pch), 10*np.log10(pup/pch)

def report(tag, ref, y, os=4):
    lo, up = aclr_db(y, os)
    ev = evm_pct(ref, y, os)
    nm = nmse_db(ref, y)
    print(f"  {tag:12s} ACLR_low={lo:7.2f} dB  ACLR_up={up:7.2f} dB   EVM={ev:7.3f} %   NMSE={nm:7.2f} dB")
    return dict(acl_low=lo, acl_up=up, evm=ev, nmse=nm)

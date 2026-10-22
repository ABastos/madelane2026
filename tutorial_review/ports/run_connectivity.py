"""Python port of the Bastos functional-connectivity tutorial (MaDeLaNe 2024/2025),
which reproduces simulations from Bastos & Schoffelen (2016, Front Syst Neurosci):

  sim_signaltonoise.m   - Granger causality under unequal SNR, and the
                          time-reversal test (Figs 7-8 of the paper)
  sim_volumeconduction.m - coherence vs imaginary coherence under field spread

FieldTrip calls are replaced by numpy equivalents:
  ft_connectivitysimulation('ar') -> MVAR simulation with lag-1/lag-2 params
  ft_freqanalysis(mtmfft, dpss)    -> multitaper cross-spectral density
  ft_connectivityanalysis(granger) -> Wilson spectral factorisation + Geweke
  ft_mvaranalysis + mvar freq      -> OLS VAR fit -> parametric spectral Granger
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import dpss

OUT = Path(__file__).resolve().parent.parent / "outputs" / "connectivity"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(3)


def simulate_ar(params, noisecov, ntrials, nsamp, absnoise=None, burn=200):
    """x_t = sum_k A_k x_{t-k} + e_t ; params[:, :, k] = A_{k+1} (row = target)."""
    nsig, _, order = params.shape
    L = np.linalg.cholesky(noisecov)
    out = np.empty((ntrials, nsig, nsamp))
    for tr in range(ntrials):
        e = L @ rng.standard_normal((nsig, nsamp + burn))
        x = np.zeros((nsig, nsamp + burn))
        for t in range(order, nsamp + burn):
            x[:, t] = sum(params[:, :, k] @ x[:, t - k - 1] for k in range(order)) + e[:, t]
        out[tr] = x[:, burn:]
    if absnoise is not None:
        out += np.asarray(absnoise)[None, :, None] * rng.standard_normal(out.shape)
    return out - out.mean(2, keepdims=True)          # blc = 'yes'


def mt_csd(data, fs, W, fmax):
    """Multitaper cross-spectral density, averaged over trials and tapers."""
    ntr, nch, n = data.shape
    K = int(np.floor(2 * n * W / fs - 1))
    tap = dpss(n, n * W / fs, K)
    f = np.fft.rfftfreq(n, 1 / fs)
    sel = f <= fmax
    X = np.fft.rfft(data[:, None] * tap[None, :, None], axis=3)[..., sel]   # tr x tap x ch x f
    S = np.einsum("tkif,tkjf->ijf", X, X.conj()) / (ntr * K)
    return f[sel], S


def wilson(S_onesided, tol=1e-10, maxit=500):
    """Wilson (1972) minimum-phase factorisation as in FieldTrip's sfactorization_wilson."""
    m, _, nf = S_onesided.shape
    S = np.concatenate([S_onesided, np.conj(S_onesided[:, :, -2:0:-1])], axis=2)  # two-sided
    N = S.shape[2]
    gam = np.real(np.fft.ifft(S, axis=2))
    h = np.linalg.cholesky(gam[:, :, 0]).T            # upper triangular, h' h = gam0
    psi = np.repeat(h[:, :, None], N, axis=2).astype(complex)
    I = np.eye(m)
    for _ in range(maxit):
        g = np.empty_like(psi)
        for k in range(N):
            pinv = np.linalg.inv(psi[:, :, k])
            g[:, :, k] = pinv @ S[:, :, k] @ pinv.conj().T + I
        beta = np.fft.ifft(g, axis=2)
        beta[:, :, 0] = np.triu(0.5 * beta[:, :, 0])
        beta[:, :, N // 2 + 1:] = 0
        gp = np.fft.fft(beta, axis=2)
        psi_new = np.einsum("ijf,jkf->ikf", psi, gp)
        if np.max(np.abs(psi_new - psi)) < tol:
            psi = psi_new
            break
        psi = psi_new
    A0 = np.real(np.fft.ifft(psi, axis=2))[:, :, 0]
    A0inv = np.linalg.inv(A0)
    H = np.einsum("ijf,jk->ikf", psi, A0inv)[:, :, :nf]
    Z = A0 @ A0.T
    return H, Z, S[:, :, :nf]


def geweke(H, Z, S):
    """Spectral Granger. Returns G[i, j, f] = influence j -> i."""
    m = H.shape[0]
    G = np.zeros((m, m, H.shape[2]))
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            Sii = np.real(S[i, i])
            num = (Z[j, j] - Z[i, j] ** 2 / Z[i, i]) * np.abs(H[i, j]) ** 2
            G[i, j] = np.log(Sii / (Sii - num))
    return G


def granger_nonparametric(data, fs, W=4, fmax=100, reverse=False):
    f, S = mt_csd(data, fs, W, fmax)
    if reverse:
        S = np.conj(S)
    H, Z, S = wilson(S)
    return f, geweke(H, Z, S), S


def granger_parametric(data, fs, order=2, freqs=np.arange(0, 101)):
    ntr, m, n = data.shape
    Y = np.concatenate([d[:, order:] for d in data], axis=1)
    X = np.concatenate([np.vstack([d[:, order - k - 1:n - k - 1] for k in range(order)]) for d in data], axis=1)
    B = Y @ X.T @ np.linalg.inv(X @ X.T)
    A = [B[:, k * m:(k + 1) * m] for k in range(order)]
    E = Y - B @ X
    Z = E @ E.T / E.shape[1]
    H = np.empty((m, m, len(freqs)), complex)
    for i, fr in enumerate(freqs):
        Af = np.eye(m) - sum(A[k] * np.exp(-2j * np.pi * fr * (k + 1) / fs) for k in range(order))
        H[:, :, i] = np.linalg.inv(Af)
    S = np.einsum("ijf,jk,lkf->ilf", H, Z, H.conj())
    return freqs, geweke(H, Z, S)


def coherency(S):
    d = np.sqrt(np.real(np.diagonal(S)))
    return S / np.outer(d, d)


def plot_gc(ax, f, G, title, legend=True):
    ax.plot(f, G[1, 0], color="#2a78d6", lw=2, label="1 → 2")
    ax.plot(f, G[0, 1], color="#eb6834", lw=2, label="2 → 1")
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Granger")
    if legend:
        ax.legend(frameon=False, fontsize=8)


def run_snr():
    fs, n, ntr = 200, 100, 1000
    P = np.zeros((2, 2, 2))
    P[:, :, 0] = [[0.55, 0.025], [0.025, 0.55]]
    P[:, :, 1] = [[-0.8, -0.1], [-0.1, -0.8]]
    C = np.array([[1, .3], [.3, 1]])
    clean = simulate_ar(P, C, ntr, n)
    noisy = simulate_ar(P, C, ntr, n, absnoise=[1, 0])
    f, G_clean, S_clean = granger_nonparametric(clean, fs)
    fp, Gp_clean = granger_parametric(clean, fs)
    _, G_noisy, S_noisy = granger_nonparametric(noisy, fs)
    fp, Gp_noisy = granger_parametric(noisy, fs)
    _, G_noisy_rev, _ = granger_nonparametric(noisy, fs, reverse=True)
    # strong, truly unidirectional 2 -> 1
    P2 = P.copy(); P2[1, 0, :] = 0
    uni = simulate_ar(P2, C, ntr, n)
    _, G_uni, _ = granger_nonparametric(uni, fs)
    _, G_uni_rev, _ = granger_nonparametric(uni, fs, reverse=True)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    axs[0, 0].plot(f, np.real(S_noisy[0, 0]), color="#2a78d6", lw=2, label="chan 1 (+ noise)")
    axs[0, 0].plot(f, np.real(S_noisy[1, 1]), color="#eb6834", lw=2, label="chan 2")
    axs[0, 0].set_title("Power, case 2 (extra noise on chan 1)", fontsize=9.5)
    axs[0, 0].set_xlabel("Frequency (Hz)"); axs[0, 0].legend(frameon=False, fontsize=8)
    plot_gc(axs[0, 1], f, G_clean, "Case 1: symmetric coupling, equal SNR\n(non-parametric GC)")
    axs[0, 1].plot(fp, Gp_clean[1, 0], color="#2a78d6", ls=":", lw=1.2)
    axs[0, 1].plot(fp, Gp_clean[0, 1], color="#eb6834", ls=":", lw=1.2)
    plot_gc(axs[0, 2], f, G_noisy, "Case 2: same coupling, chan 1 noisier\n→ spurious asymmetry (dotted: parametric)")
    axs[0, 2].plot(fp, Gp_noisy[1, 0], color="#2a78d6", ls=":", lw=1.2)
    axs[0, 2].plot(fp, Gp_noisy[0, 1], color="#eb6834", ls=":", lw=1.2)
    plot_gc(axs[1, 0], f, G_noisy_rev, "Case 2 time-reversed:\nasymmetry does NOT flip → SNR artefact")
    plot_gc(axs[1, 1], f, G_uni, "Strong case: true 2 → 1 only\n(forward time)")
    plot_gc(axs[1, 2], f, G_uni_rev, "Strong case time-reversed:\nasymmetry flips → genuine directed influence")
    fig.suptitle("Granger causality & the time-reversal test  [tutorial's own AR simulations, sim_signaltonoise.m]",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "granger_snr_timereversal.png", dpi=160); plt.close(fig)

    k40 = int(np.argmin(np.abs(f - 40)))
    return {"peak_freq_hz": float(f[np.argmax(G_clean[1, 0] + G_clean[0, 1])]),
            "case1_gc_40hz": {"1->2": float(G_clean[1, 0, k40]), "2->1": float(G_clean[0, 1, k40])},
            "case2_noisy_gc_40hz": {"1->2": float(G_noisy[1, 0, k40]), "2->1": float(G_noisy[0, 1, k40])},
            "case2_reversed_gc_40hz": {"1->2": float(G_noisy_rev[1, 0, k40]), "2->1": float(G_noisy_rev[0, 1, k40])},
            "unidirectional_gc_40hz": {"1->2": float(G_uni[1, 0, k40]), "2->1": float(G_uni[0, 1, k40])},
            "unidirectional_reversed_gc_40hz": {"1->2": float(G_uni_rev[1, 0, k40]), "2->1": float(G_uni_rev[0, 1, k40])}}


def run_volume_conduction():
    fs, n, ntr, ns = 200, 200, 500, 50
    P = np.zeros((ns, ns, 2))
    P[:, :, 0] = np.diag(np.full(ns, .55)); P[:, :, 1] = np.diag(np.full(ns, -.8))
    C = np.diag(np.full(ns, .05)); C[15, 15] = C[34, 34] = 1
    hw = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, 32) / 32))          # MATLAB hanning(31)
    tra = np.array([[hw[i - j + 15] if 0 <= i - j + 15 < 31 else 0 for j in range(ns)] for i in range(ns)])
    mix = lambda d: np.einsum("ij,tjn->tin", tra, d)
    f, S0 = mt_csd(mix(simulate_ar(P, C, ntr, n)), fs, 5, 100)
    P[15, 34, 0], P[15, 34, 1] = .25, -.1                                  # source 35 drives 16
    _, S1 = mt_csd(mix(simulate_ar(P, C, ntr, n)), fs, 5, 100)
    k = int(np.argmin(np.abs(f - 40)))
    c0, c1 = coherency(S0[:, :, k]), coherency(S1[:, :, k])

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    panels = [(np.abs(c0), "coherence — NO coupling", 0, 1),
              (np.abs(np.imag(c0)), "imag(coh) — NO coupling", 0, .05),
              (np.abs(c1), "coherence — 35 → 16 coupled", 0, 1),
              (np.abs(np.imag(c1)), "imag(coh) — 35 → 16 coupled", 0, .5)]
    for ax, (M, t, lo, hi) in zip(axs, panels):
        im = ax.imshow(M, cmap="viridis", vmin=lo, vmax=hi, origin="upper",
                       extent=[.5, ns + .5, ns + .5, .5])
        ax.plot([16, 35], [16, 35], "o", mfc="white", mec="#0b0b0b", ms=6)
        ax.set_title(t + "  (40 Hz)", fontsize=9.5); ax.set_xlabel("Sensor"); ax.set_ylabel("Sensor")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("Volume conduction: coherence leaks, imaginary coherence does not  [sim_volumeconduction.m]",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "volume_conduction_coh_vs_icoh.png", dpi=160); plt.close(fig)
    return {"coh_16_17_uncoupled": float(np.abs(c0[15, 16])),
            "icoh_16_17_uncoupled": float(np.abs(np.imag(c0[15, 16]))),
            "coh_16_35_uncoupled": float(np.abs(c0[15, 34])),
            "coh_16_35_coupled": float(np.abs(c1[15, 34])),
            "icoh_16_35_coupled": float(np.abs(np.imag(c1[15, 34])))}


if __name__ == "__main__":
    res = {"granger_snr": run_snr(), "volume_conduction": run_volume_conduction()}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

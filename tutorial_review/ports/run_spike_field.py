"""Python port of the Vinck spike-field coherence tutorial (MaDeLaNe 2024/2025):
create_data.m, compute_avg_ppc.m and main_tutorial_for_tutors_2025.m.

Uses the tutorial's own synthetic-data path (`[lfp, dataspike] = create_data()`):
a 40 Hz damped AR(2) 'LFP' drives 100 inhomogeneous-Bernoulli neurons whose
spike probability is proportional to the LFP. FieldTrip calls are replaced by
equivalent numpy code:
  ft_spiketriggeredspectrum_convol -> Hann-tapered wavelets, 5 cycles per freq
  ft_spiketriggeredspectrum_stat   -> ppc1 / plv / Rayleigh
  ft_spiketriggeredaverage         -> +-100 ms STA
  ft_freqanalysis + coh            -> multitaper (dpss, 8 Hz smoothing) coherence
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import dpss

OUT = Path(__file__).resolve().parent.parent / "outputs" / "spike_field"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(7)


def create_data(f=40, eigenvalue=0.94, Fs=1000, T=100, nNeurons=100):
    omega = 2 * np.pi * f / Fs
    a2 = -eigenvalue ** 2
    a1 = 4 * a2 * np.cos(omega) / (a2 - 1)   # as written in create_data.m
    x = lfilter([1], [1, -a1, -a2], rng.standard_normal(T * Fs))
    x = x - x.min(); x = x / x.max()
    lfp = x.reshape(T, Fs)                  # 100 trials x 1 s
    rates = rng.exponential(5, nNeurons)
    spikes = np.zeros((nNeurons, T, Fs), bool)
    for i, Fr in enumerate(rates):
        p = Fr / Fs * x
        while True:
            S = p > rng.random(x.size)
            if S.sum() >= 3:
                break
        spikes[i] = S.reshape(T, Fs)
    return lfp, spikes, Fs


def sts_convol(lfp, spikes, Fs, foi):
    """Spike-triggered spectrum: complex LFP coefficient at each spike, per freq."""
    nN, nT, nS = spikes.shape
    out = [dict(trial=[], coef=[]) for _ in range(nN)]
    conv = np.empty((len(foi), nT, nS), complex)
    for k, f in enumerate(foi):
        n = int(round(5 / f * Fs))
        tt = (np.arange(n) - (n - 1) / 2) / Fs
        w = np.hanning(n) * np.exp(-2j * np.pi * f * tt)
        w /= np.abs(w).sum()
        for tr in range(nT):
            c = np.convolve(lfp[tr] - lfp[tr].mean(), w[::-1], mode="same")
            c[: n // 2] = np.nan; c[nS - n // 2:] = np.nan    # window must fit in trial
            conv[k, tr] = c
    for i in range(nN):
        tr_idx, s_idx = np.nonzero(spikes[i])
        out[i]["trial"] = tr_idx
        out[i]["coef"] = conv[:, tr_idx, s_idx].T          # nspikes x nfreq
    return out


def ppc1(coef, trial):
    """FieldTrip ppc1: pairwise phase consistency over spike pairs from different trials."""
    z = coef / np.abs(coef)
    res = np.full(z.shape[1], np.nan)
    nsp = np.zeros(z.shape[1], int)
    for k in range(z.shape[1]):
        ok = ~np.isnan(z[:, k])
        zk, tk = z[ok, k], trial[ok]
        nsp[k] = ok.sum()
        if nsp[k] < 2:
            continue
        S = np.array([zk[tk == m].sum() for m in np.unique(tk)])
        n_m = np.array([(tk == m).sum() for m in np.unique(tk)])
        den = nsp[k] ** 2 - (n_m ** 2).sum()
        if den > 0:
            res[k] = (np.abs(S.sum()) ** 2 - (np.abs(S) ** 2).sum()) / den
    return res, nsp


def compute_avg_ppc(n_spikes, ppc_all):
    w = np.nansum(ppc_all * n_spikes / np.nansum(n_spikes, 0), 0)
    uw = np.nanmean(ppc_all, 0)
    th = np.where(n_spikes < 50, np.nan, ppc_all)
    return w, uw, np.nanmean(th, 0)


def mt_coherence(x, y, Fs, W=8, fmax=100):
    n = x.shape[1]
    K = int(2 * n * W / Fs - 1)
    tap = dpss(n, n * W / Fs, K)
    fr = np.fft.rfftfreq(n, 1 / Fs)
    sel = (fr >= 1) & (fr <= fmax)
    X = np.fft.rfft(tap[None] * (x - x.mean(1, keepdims=True))[:, None], axis=2)[..., sel]
    Y = np.fft.rfft(tap[None] * (y - y.mean(1, keepdims=True))[:, None], axis=2)[..., sel]
    Sxy = (X * Y.conj()).mean((0, 1)); Sxx = (np.abs(X) ** 2).mean((0, 1)); Syy = (np.abs(Y) ** 2).mean((0, 1))
    return fr[sel], np.abs(Sxy) / np.sqrt(Sxx * Syy)


def main():
    lfp, spikes, Fs = create_data()
    foi = np.arange(10, 101)
    sts = sts_convol(lfp, spikes, Fs, foi)
    nN = len(sts)
    ppc_all = np.full((nN, len(foi)), np.nan)
    n_spk = np.full((nN, len(foi)), np.nan)
    for i in range(nN):
        if len(sts[i]["trial"]) < 3:
            continue
        ppc_all[i], n_spk[i] = ppc1(sts[i]["coef"], sts[i]["trial"])
    w, uw, th = compute_avg_ppc(n_spk, ppc_all)

    # STA (+-100 ms)
    lag = np.arange(-100, 101)
    sta = np.full((nN, len(lag)), np.nan)
    for i in range(nN):
        tr, s = np.nonzero(spikes[i])
        ok = (s >= 100) & (s < Fs - 100)
        if ok.any():
            sta[i] = np.mean([lfp[t, j - 100:j + 101] for t, j in zip(tr[ok], s[ok])], 0)

    # Example neuron with most spikes: phase histogram, STA, PPC, PLV
    i = int(np.nanargmax(n_spk[:, 0]))
    k40 = int(np.where(foi == 40)[0][0])
    z = sts[i]["coef"][:, k40]; z = z[~np.isnan(z)]
    ph = np.angle(z)
    zall = sts[i]["coef"] / np.abs(sts[i]["coef"])
    plv = np.abs(np.nanmean(zall, 0))
    n = len(z); R = np.abs(np.mean(z / np.abs(z))) * n
    ral_p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    fig = plt.figure(figsize=(10, 6.6))
    ax = fig.add_subplot(2, 2, 1, projection="polar")
    ax.hist(ph, bins=20, color="#2a78d6", edgecolor="white")
    ax.set_title(f"Neuron {i + 1}: spike phases at 40 Hz\n{n} spikes, Rayleigh p = {ral_p:.1e}", fontsize=9)
    ax.set_yticklabels([])
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(lag, sta[i], color="#0b0b0b"); ax.set_xlabel("Lag from spike (ms)"); ax.set_ylabel("STA (a.u.)")
    ax.set_title("Spike-triggered average", fontsize=10)
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(foi, ppc_all[i], color="#0b0b0b"); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PPC1")
    ax.set_title("Pairwise phase consistency", fontsize=10)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(foi, plv, color="#0b0b0b"); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PLV")
    ax.set_title("Phase-locking value (biased by spike count)", fontsize=10)
    fig.suptitle("Part 5: single neuron  [tutorial's synthetic data, create_data.m]", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "part5_single_neuron.png", dpi=160); plt.close(fig)

    # Part 4 + Part 6 + Part 8 summary figure
    sfc_f, sfc = [], []
    for j in range(nN):
        f_, c_ = mt_coherence(spikes[j].astype(float), lfp, Fs)
        sfc.append(c_); sfc_f = f_
    sfc = np.nanmean(sfc, 0)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    order = np.argsort(-np.nansum(spikes, (1, 2)))
    im = axs[0].imshow(sta[order], aspect="auto", cmap="viridis", extent=[lag[0], lag[-1], nN, 1])
    axs[0].set_xlabel("Lag from spike (ms)"); axs[0].set_ylabel("Neuron (sorted by spike count)")
    axs[0].set_title("Part 4: STA, all neurons", fontsize=10); fig.colorbar(im, ax=axs[0])
    for y, c, lab in [(w, "#1baf7a", "weighted"), (uw, "#e34948", "unweighted"), (th, "#2a78d6", "≥50 spikes")]:
        axs[1].plot(foi, y, color=c, lw=2, label=lab)
    axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("PPC1"); axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("Part 6: population PPC", fontsize=10)
    axs[2].plot(sfc_f, sfc, color="#6250d6", lw=2)
    axs[2].set_xlabel("Frequency (Hz)"); axs[2].set_ylabel("Coherence")
    axs[2].set_title("Part 8: spike-field coherence (mean)", fontsize=10)
    fig.suptitle("Vinck spike-field tutorial  [tutorial's synthetic data, 40 Hz generator]", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "population_ppc_sfc.png", dpi=160); plt.close(fig)

    summary = {"n_neurons": nN, "total_spikes": int(spikes.sum()),
               "weighted_ppc_peak_hz": int(foi[np.nanargmax(w)]), "weighted_ppc_peak": float(np.nanmax(w)),
               "thresh_ppc_peak_hz": int(foi[np.nanargmax(th)]),
               "sfc_peak_hz": float(sfc_f[np.nanargmax(sfc)]), "sfc_peak": float(np.nanmax(sfc)),
               "example_neuron": i + 1, "example_rayleigh_p_40hz": float(ral_p),
               "n_neurons_ge50_spikes": int((n_spk[:, 0] >= 50).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Run Spectrolaminar_Tutorial_0.m + _1.m (MaDeLaNe 2026) and CSDanalysis.m
(MaDeLaNe 2024, Maier) in Python.

The tutorial's recording (TrialCutData_AreaMT.mat, 16 ch, 150 um, 1 kHz) could
not be downloaded from Box in this environment, so this script builds a
SIMULATED laminar dataset with the same layout and a known ground truth:
  * gamma (60-100 Hz) power peaks superficially (L2/3, ch 3-4)
  * alpha/beta (10-25 Hz) power peaks deep (L5/6, ch 12-13)
  * a stimulus-evoked current sink in L4 (ch 8) at 40-80 ms
Pass --mat path/to/TrialCutData_AreaMT.mat to run on the real file instead.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

from vflip2 import VFLIP2, hanning_matlab

OUT = Path(__file__).resolve().parent.parent / "outputs" / "spectrolaminar"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def ar2(n, f, r, fs, rng):
    th = 2 * np.pi * f / fs
    return lfilter([1], [1, -2 * r * np.cos(th), r ** 2], rng.standard_normal(n))


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    return np.fft.irfft(X / f ** 0.9, n)


def forward_csd_to_lfp(csd, z, sigma=0.4, R=0.5):
    # Pettersen et al. disc model: phi(z) = 1/(2 sigma) * sum C(z') (sqrt((z-z')^2+R^2) - |z-z'|) dz
    dz = z[1] - z[0]
    K = (np.sqrt((z[:, None] - z[None]) ** 2 + R ** 2) - np.abs(z[:, None] - z[None])) / (2 * sigma)
    return K @ csd * dz


def simulate(seed=1, n_trials=120, nchan=16, intdist=0.15, fs=1000):
    rng = np.random.default_rng(seed)
    ch = np.arange(1, nchan + 1)
    z = (ch - 1) * intdist
    w_gamma = np.exp(-0.5 * ((ch - 3.5) / 2.2) ** 2)
    w_beta = np.exp(-0.5 * ((ch - 12.5) / 2.5) ** 2)
    smooth = np.exp(-0.5 * ((ch[:, None] - ch[None]) / 1.0) ** 2)
    smooth /= smooth.sum(1, keepdims=True)
    # evoked CSD (current sink negative): L4 sink -> L2/3 + L5 sinks later
    t_ev = np.arange(0, 0.4, 1 / fs)
    def blob(c, t0, tw, cw):
        return np.exp(-0.5 * ((ch - c) / cw) ** 2)[:, None] * np.exp(-0.5 * ((t_ev - t0) / tw) ** 2)[None]
    csd_ev = (-blob(8, .055, .012, 1.0) + .6 * blob(5, .06, .015, 1.2) + .6 * blob(11, .06, .015, 1.2)
              - .7 * blob(4, .11, .025, 1.3) - .5 * blob(12, .12, .03, 1.3) + .4 * blob(8, .12, .03, 1.5))
    lfp_ev = forward_csd_to_lfp(csd_ev * 150, z)
    trials, onsets = [], []
    for _ in range(n_trials):
        n = int(rng.integers(1800, 3200))
        g = np.stack([ar2(n, rng.uniform(70, 85), .90, fs, rng) for _ in ch])
        b = np.stack([ar2(n, rng.uniform(14, 18), .965, fs, rng) for _ in ch])
        x = (smooth @ (w_gamma[:, None] * g * 1.6 + w_beta[:, None] * b * 1.0)
             + (0.6 + 0.02 * ch)[:, None] * pink(n, rng)[None] * 12
             + 0.15 * rng.standard_normal((nchan, n)))
        on = int(rng.integers(300, n - 500))
        x[:, on:on + len(t_ev)] += lfp_ev
        trials.append(x * 40)  # ~uV scale
        onsets.append(on)
    return dict(trials=trials, onsets=onsets, fsample=fs, intdist=intdist,
                csd_true=csd_ev, t_ev=t_ev)


def load_mat(path):
    from scipy.io import loadmat
    m = loadmat(path, simplify_cells=True)
    ds = m["DataStruct"]
    return dict(trials=[np.asarray(t, float) for t in ds["trial"]], onsets=None,
                fsample=float(ds["fsample"]), intdist=float(m["intDist"]))


def tutorial0_power(trials, fs):
    # Steps 8-9 of Spectrolaminar_Tutorial_0.m
    foi = np.arange(1, 151)
    tr = [t - t.mean(1, keepdims=True) for t in trials]
    tr = [t * (hanning_matlab(t.shape[1]) / np.linalg.norm(hanning_matlab(t.shape[1]))) for t in tr]
    npad = int(2 ** np.ceil(np.log2(max(t.shape[1] for t in tr))))
    res = fs / npad
    idx = np.unique(np.round(foi / res).astype(int) + 1)
    fz = (idx - 1) * res
    P = []
    for t in tr:
        X = np.zeros((t.shape[0], npad)); X[:, :t.shape[1]] = t
        F = np.fft.fft(X, axis=1)[:, idx - 1] * np.sqrt(2) * np.sqrt(1 / npad)
        P.append(np.abs(F) ** 2)
    return fz, np.mean(P, axis=0)


def csd_maier(epochs, dis, sig=0.4, sep=2):
    # CSDanalysis.m: -sigma * second spatial difference with channel separation `sep`
    nch = epochs.shape[1]
    out = np.zeros((epochs.shape[0], nch - 2 * sep, epochs.shape[2]))
    for c in range(sep, nch - sep):
        out[:, c - sep] = -sig * (epochs[:, c - sep] - 2 * epochs[:, c] + epochs[:, c + sep]) / (dis * sep) ** 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", help="TrialCutData_AreaMT.mat (real data)")
    a = ap.parse_args()
    D = load_mat(a.mat) if a.mat else simulate()
    label = "real MT recording" if a.mat else "SIMULATED laminar data"
    trials, fs, intdist = D["trials"], D["fsample"], D["intdist"]
    nch = trials[0].shape[0]
    ch = np.arange(1, nch + 1)

    # Step 6: raw voltage, all channels, one trial
    fig, ax = plt.subplots(figsize=(7, 4.4))
    t = trials[15]
    seg = slice(D["onsets"][15] - 100, D["onsets"][15] + 200) if D["onsets"] else slice(6100, 6400)
    for c in range(nch):
        ax.plot(np.arange(seg.start, seg.stop), t[c, seg] + (nch - 1 - c) * 500, color="#2a78d6", lw=.8)
    ax.set_yticks([(nch - 1 - c) * 500 for c in range(0, nch, 3)], [str(c + 1) for c in range(0, nch, 3)])
    ax.set_ylabel("Channel (1 = superficial)"); ax.set_xlabel("Sample (ms)")
    ax.set_title(f"Step 6: raw LFP, all channels, trial 16  [{label}]", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "step6_raw_traces.png", dpi=160); plt.close(fig)

    # Steps 8-13: absolute and relative laminar power
    fz, meanPow = tutorial0_power(trials, fs)
    relpow = meanPow / meanPow.max(axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, M, ttl, cmap in [(axs[0], np.log10(meanPow), "Step 10: absolute power (log10)", "magma"),
                             (axs[1], relpow, "Step 12: relative power (÷ max over depth)", "viridis")]:
        im = ax.imshow(M, aspect="auto", cmap=cmap, extent=[fz[0], fz[-1], nch + .5, .5])
        fig.colorbar(im, ax=ax); ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Channel (1 = superficial)")
    fig.suptitle(f"Laminar power spectrum  [{label}]", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "step10_12_power_maps.png", dpi=160); plt.close(fig)

    bands = {"Delta-Theta (1-6 Hz)": (1, 6), "Alpha-Beta (10-20 Hz)": (10, 20), "Gamma (70-100 Hz)": (70, 100)}
    colors = ["#52514e", "#2a78d6", "#e34948"]
    fig, ax = plt.subplots(figsize=(4.4, 4.6))
    prof = {}
    for (nm, (lo, hi)), c in zip(bands.items(), colors):
        p = relpow[:, (fz >= lo) & (fz <= hi)].mean(1)
        prof[nm] = p.round(3).tolist()
        ax.plot(p, ch, color=c, lw=2, label=nm)
    ax.set_ylim(nch + .5, .5); ax.set_xlabel("Relative power"); ax.set_ylabel("Channel (1 = superficial)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Step 13: laminar band profiles", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "step13_band_profiles.png", dpi=160); plt.close(fig)

    # Tutorial_1: vFLIP2 on trial-cut data
    flip = VFLIP2(trials, intdist=intdist, data_type="raw_cut", fsample=fs)
    R = flip.Results
    if R is not None:
        fig = flip.plot_result(title_prefix=f"vFLIP2 Results [{label}].")
        fig.savefig(OUT / "tutorial1_vflip2_result.png", dpi=160); plt.close(fig)

    # CSD (Maier 2024 tutorial) on stimulus-locked epochs
    csd_summary = None
    if D["onsets"]:
        pre, post = 100, 300
        ep = np.stack([tr[:, o - pre:o + post].T for tr, o in zip(trials, D["onsets"])], axis=2)
        ep = ep - ep[:pre].mean(0, keepdims=True)
        csd = csd_maier(ep / 1000.0, dis=intdist).mean(2)  # uV -> mV
        tt = np.arange(-pre, post)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
        erp = ep.mean(2)
        for c in range(nch):
            axs[0].plot(tt, erp[:, c] - c * 150, color="#2a78d6", lw=.9)
        axs[0].set_yticks([-c * 150 for c in range(0, nch, 3)], [str(c + 1) for c in range(0, nch, 3)])
        axs[0].axvline(0, color="#52514e", lw=.8, ls="--")
        axs[0].set_xlabel("Time from stimulus (ms)"); axs[0].set_ylabel("Channel")
        axs[0].set_title("Stimulus-locked LFP (ERP)", fontsize=10)
        v = np.abs(csd).max()
        im = axs[1].imshow(csd.T, aspect="auto", cmap="RdBu", vmin=-v, vmax=v,
                           extent=[tt[0], tt[-1], nch - 2 + .5, 2 + .5])
        axs[1].axvline(0, color="k", lw=.8, ls="--")
        axs[1].set_xlim(-50, 250)
        cb = fig.colorbar(im, ax=axs[1]); cb.set_label("CSD (sink < 0 = red)")
        axs[1].set_xlabel("Time from stimulus (ms)"); axs[1].set_ylabel("Channel")
        axs[1].set_title("CSD (Maier tutorial: sep = 2, σ = 0.4)", fontsize=10)
        fig.suptitle(f"Current source density  [{label}]", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / "maier_csd.png", dpi=160); plt.close(fig)
        win = (tt >= 30) & (tt <= 80)
        sink_ch = int(np.argmin(csd[win].mean(0))) + 3
        csd_summary = {"earliest_sink_channel_30_80ms": sink_ch, "planted_L4_sink_channel": 8}

    summary = {"data": label, "n_trials": len(trials), "n_channels": nch, "intdist_mm": intdist,
               "fsample": fs, "band_profiles": prof,
               "vflip2": None if R is None else {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
                                                  for k, v in R.items()},
               "csd": csd_summary}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("data", "vflip2", "csd")}, indent=2))


if __name__ == "__main__":
    main()

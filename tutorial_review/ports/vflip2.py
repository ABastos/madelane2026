"""Python port of vFLIP2.m (MaDeLaNe 2026, Multi-Laminar Spectral Analysis).

Line-by-line translation of the MATLAB class: same PSD estimate, same
proximal/distal channel search, same goodness (G) and omega scoring, same
physiological checks. MATLAB 1-based channel indices are kept in the
Results dict so outputs are directly comparable with the MATLAB tool.
"""
import numpy as np
from scipy import stats

# O'Kusky & Colonnier (1982) laminar thicknesses (mm), macaque V1
LAMINAE_MM = {"I": .1229, "II-III": .3969, "IVA": .1271, "IVB": .2114,
              "IVC": .2475, "V": .2263, "VI": .2602}


def hanning_matlab(n):
    # MATLAB hanning(n): symmetric, excludes the zero end-points
    k = np.arange(1, n + 1)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n + 1)))


def movmean(x, w=5):
    # MATLAB smoothdata(x,'movmean',w): centred window, shrinks at the edges
    n = len(x)
    half_lo, half_hi = (w - 1) // 2, w // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = x[max(0, i - half_lo):min(n, i + half_hi + 1)].mean()
    return out


def longest_true_run(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    if len(starts) == 0:
        return np.array([], dtype=int)
    i = np.argmax(ends - starts + 1)
    return np.arange(starts[i], ends[i] + 1)


def compute_psd_hanning(trials, fsample):
    """trials: list of [nchan x nsamp] arrays. Returns mean power [nchan x 150]."""
    foi = np.arange(1, 151)
    max_n = max(t.shape[1] for t in trials)
    padding = int(2 ** np.ceil(np.log2(max_n)))
    pad = padding / fsample
    endnsample = int(round(pad * fsample))
    fboi = np.round(foi * pad) + 1
    foi = (fboi - 1) / pad
    freqboi = np.unique(np.round(foi / (fsample / endnsample)) + 1).astype(int)
    pow_all = []
    for dat in trials:
        dat = np.asarray(dat, float)
        n = dat.shape[1]
        tap = hanning_matlab(n)
        tap /= np.linalg.norm(tap)
        dat = dat - dat.mean(axis=1, keepdims=True)  # regression on a constant
        x = np.zeros((dat.shape[0], endnsample))
        x[:, :n] = dat * tap
        dum = np.fft.fft(x, axis=1)[:, freqboi - 1]
        dum *= np.sqrt(2 / endnsample)
        pow_all.append(np.abs(dum) ** 2)
    return np.mean(pow_all, axis=0)


def split_into_trials(data, fsample):
    spt = int(fsample)
    nt = data.shape[1] // spt
    return [data[:, i * spt:(i + 1) * spt] for i in range(nt)]


def band_regress(band):
    n = len(band)
    t = np.arange(1, n + 1)
    X = np.column_stack([t ** 2, t, np.ones(n)])
    p, *_ = np.linalg.lstsq(X, band, rcond=None)
    yhat = X @ p
    ss_res = np.sum((band - yhat) ** 2)
    ss_tot = np.sum((band - band.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df1, df2 = 2, n - 3
    if df2 <= 0 or r2 >= 1:
        pval = 0.0 if r2 >= 1 else 1.0
    else:
        F = (r2 / df1) / ((1 - r2) / df2)
        pval = stats.f.sf(F, df1, df2)
    mid = int(np.floor(n / 2 + 0.5))  # MATLAB round
    slope = 2 * p[0] * mid + p[1]
    return slope, r2, pval


class VFLIP2:
    def __init__(self, data, intdist, data_type="psd", fsample=None,
                 orientation="both", layer4_thickness=None, omega_cut=6.0):
        self.intdist = intdist
        self.step = max(1, int(np.floor(0.1 / intdist + 0.5)))
        self.minrange_s = int(np.ceil(.3 / intdist))
        self.omega_cut = omega_cut
        if data_type == "psd":
            self.pow = np.asarray(data, float)
        elif data_type == "raw":
            self.pow = compute_psd_hanning(split_into_trials(data, fsample), fsample)
        elif data_type == "raw_cut":
            self.pow = compute_psd_hanning(data, fsample)
        self.freqbinsize = 1
        good = ~np.isnan(self.pow[:, 0])
        self.startrow = int(np.argmax(good)) + 1            # 1-based
        self.endrow = self.startrow + int(good.sum()) - 1
        self.freqaxis = np.arange(1, self.pow.shape[1] + 1) * self.freqbinsize
        self.orientation1 = {"both": 0, "upright": 1, "inverted": -1}[orientation]
        l4 = layer4_thickness or (LAMINAE_MM["IVA"] + LAMINAE_MM["IVB"] + LAMINAE_MM["IVC"])
        self.minrange = int(np.ceil(l4 / intdist))
        self.Results = self._flip_it()

    # rows are 1-based inclusive, as in MATLAB
    def get_window(self, a, b):
        w = self.pow[a - 1:b, :]
        return w / w.max(axis=0)

    def get_freqbands(self, s1, s2):
        fa = self.freqaxis
        low = (fa > 4) & (fa < 70)
        high = (fa > 40) & (fa < 150)
        r1 = longest_true_run((s1 > s2) & low)
        r2 = longest_true_run((s1 < s2) & low)
        deep_f, sup_f, orient = np.array([]), np.array([]), 0
        if len(r1) >= 5 and len(r2) >= 5:
            if fa[r1].min() < fa[r2].min():
                deep_f, orient, ind_high = fa[r1], -1, (s1 < s2) & high
            else:
                deep_f, orient, ind_high = fa[r2], 1, (s1 > s2) & high
        elif len(r1) >= 5:
            deep_f, orient, ind_high = fa[r1], -1, (s1 < s2) & high
        elif len(r2) >= 5:
            deep_f, orient, ind_high = fa[r2], 1, (s1 > s2) & high
        if orient != 0:
            sup_f = fa[longest_true_run(ind_high)]
            if len(sup_f) < 20 or sup_f[-1] < 70:
                sup_f = np.array([])
        return deep_f, sup_f, orient

    def peak_check(self, band, prox, dist, minrange_s):
        n = len(band)
        locs = np.where(band == band.max())[0] + 1
        pk = locs.max() if locs.mean() > n / 2 else locs.min()
        if pk == 1:
            ok = prox == self.startrow and band[0] > band[1]
        elif pk == n:
            ok = dist == self.endrow and band[-1] > band[-2]
        else:
            ok = band[pk - 1] > band[pk] and band[pk - 1] > band[pk - 2]
        idx = int(np.argmax(band)) + 1
        return bool(ok) and (idx <= minrange_s or idx >= n - (minrange_s - 1))

    @staticmethod
    def crossover(low, high, prox, orient):
        diff = np.abs(high - low)
        b1, b2 = (high, low) if orient > 0 else (low, high)
        cands = []
        for i in range(len(low) - 2):  # 0-based i == MATLAB i+1
            if b1[i] > b2[i] and b2[i + 1] > b1[i + 1]:
                cands.append(i + 1 if diff[i] <= diff[i + 1] else i + 2)
            elif b1[i] > b2[i] and b1[i + 1] == b2[i + 1] and b2[i + 2] > b1[i + 2]:
                cands.append(i + 2)
        if not cands:
            return np.nan
        if len(cands) == 1:
            return cands[0] + prox - 1
        ratings = [diff[:c].sum() - diff[c - 1:].sum() for c in cands]
        return cands[int(np.argmax(ratings))] + prox - 1

    @staticmethod
    def goodness(low, high):
        ls, lr2, lp = band_regress(low)
        hs, hr2, hp = band_regress(high)
        g = hr2 * lr2 * (lp < .05 and hp < .05)
        sign = 1 if (ls > 0 and hs < 0) else (-1 if (ls < 0 and hs > 0) else 0)
        return g * sign

    def _flip_it(self):
        best, best_omega = None, -np.inf
        fa = self.freqaxis
        for prox in range(self.startrow, self.endrow - self.minrange + 2, self.step):
            for dist in range(prox + self.minrange, self.endrow + 1, self.step):
                rp = self.get_window(prox, dist)
                ms = abs(prox - dist) // 2
                s1 = movmean(rp[:ms].mean(axis=0))
                s2 = movmean(rp[-ms:].mean(axis=0))
                ps_dist = np.sqrt(np.sum((s1 - s2) ** 2))
                deep_f, sup_f, orient = self.get_freqbands(s1, s2)
                if len(deep_f) == 0 or len(sup_f) == 0:
                    continue
                low = rp[:, np.isin(fa, deep_f)].mean(axis=1)
                high = rp[:, np.isin(fa, sup_f)].mean(axis=1)
                band_dist = np.sqrt(np.sum((low - high) ** 2))
                g = self.goodness(low, high)
                if self.orientation1 == -1 and g > 0:
                    g = 0
                if self.orientation1 == 1 and g < 0:
                    g = 0
                ch_span = abs(prox - dist) * (self.intdist / .1)
                f_span = len(deep_f) * len(sup_f) * self.freqbinsize ** 2
                with np.errstate(divide="ignore"):
                    omega = np.log(ps_dist * band_dist * abs(g) * ch_span * f_span)
                hi_ch = int(np.argmax(high)) + prox
                lo_ch = int(np.argmax(low)) + prox
                cross = self.crossover(low, high, prox, orient)
                ok = (omega != 0
                      and self.peak_check(low, prox, dist, ms)
                      and self.peak_check(high, prox, dist, ms)
                      and abs(hi_ch - lo_ch) >= self.minrange
                      and not np.isnan(cross)
                      and ((lo_ch < cross < hi_ch) or (lo_ch > cross > hi_ch))
                      and lo_ch != cross and cross != hi_ch and lo_ch != hi_ch)
                if ok and omega > best_omega:
                    best_omega = omega
                    best = dict(goodnessvalue=g, startinglowfreq=deep_f[0],
                                endinglowfreq=deep_f[-1], startinghighfreq=sup_f[0],
                                endinghighfreq=sup_f[-1], proximalchannel=prox,
                                distalchannel=dist, lowfreqmaxchannel=lo_ch,
                                highfreqmaxchannel=hi_ch, crossoverchannel=cross,
                                omega=omega)
        if best_omega <= self.omega_cut:
            return None
        return best

    def plot_result(self, fig=None, title_prefix="vFLIP2 Results."):
        import matplotlib.pyplot as plt
        R = self.Results
        if R is None:
            raise ValueError("No Results to show (omega below cut-off)")
        fig = fig or plt.figure(figsize=(11, 4.6))
        mx = self.pow[R["proximalchannel"] - 1:R["distalchannel"], :].max(axis=0)
        rel = self.pow / mx
        nch = rel.shape[0]
        ax1 = fig.add_subplot(1, 3, (1, 2))
        im = ax1.imshow(rel, aspect="auto", cmap="viridis", vmin=.3, vmax=1,
                        extent=[self.freqaxis[0] - .5, self.freqaxis[-1] + .5, nch + .5, .5])
        for f, c in [(R["startinglowfreq"], "#2a78d6"), (R["endinglowfreq"], "#2a78d6"),
                     (R["startinghighfreq"], "#e34948"), (R["endinghighfreq"], "#e34948")]:
            ax1.axvline(f, color=c, ls="--", lw=1.2)
        cb = fig.colorbar(im, ax=ax1)
        cb.set_label("Relative power")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Channel (1 = superficial)")
        ax1.set_title("LFP relative power  (blue: low band, red: high band)", fontsize=10)
        l_i = (self.freqaxis >= R["startinglowfreq"]) & (self.freqaxis <= R["endinglowfreq"])
        h_i = (self.freqaxis >= R["startinghighfreq"]) & (self.freqaxis <= R["endinghighfreq"])
        ax2 = fig.add_subplot(1, 3, 3)
        ch = np.arange(1, nch + 1)
        ax2.plot(rel[:, l_i].mean(1), ch, color="#2a78d6", lw=2, label="alpha/beta band")
        ax2.plot(rel[:, h_i].mean(1), ch, color="#e34948", lw=2, label="gamma band")
        ax2.axhspan(R["proximalchannel"], R["distalchannel"], xmin=.95, color="#eda100", alpha=.3)
        for key, lab in [("crossoverchannel", "crossover"),
                         ("lowfreqmaxchannel", "alpha/beta peak"),
                         ("highfreqmaxchannel", "gamma peak")]:
            for ax in (ax1, ax2):
                ax.axhline(R[key], color="k", lw=.7, ls="-.")
            ax2.text(0.02, R[key] - .15, lab, fontsize=8, color="#52514e")
        ax2.set_ylim(nch + .5, .5)
        ax2.set_xlim(0, 1.02)
        ax2.set_xlabel("Relative power")
        ax2.legend(loc="lower right", fontsize=8, frameon=False)
        ax2.set_title("Band-averaged profile", fontsize=10)
        fig.suptitle(f"{title_prefix}  G = {R['goodnessvalue']:.3f},  "
                     f"ω = {R['omega']:.2f}", fontsize=12)
        fig.tight_layout()
        return fig

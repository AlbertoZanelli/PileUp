import sys, csv, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os; BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BASE)
import src.analysis as an, src.simulation as sim
CH, WP = 34, 15
WD = f"{BASE}/m205_results_wiener_APsimfit10000led_npsclean_swna1_hist"
A = {(int(r["channel"]), int(r["wp"])): float(r["signal_amp"]) for r in csv.DictReader(open(glob.glob(f"{WD}/BI_results_*.csv")[0]))}[(CH, WP)]
gen = np.load(f"{BASE}/residual_scan_bessel/fits_octopus/bestfit_ch{CH}_wp{WP}.npy")
nps = np.load(f"{BASE}/m205_NPS_clean/ch{CH}/nps_ch{CH}_wp{WP}.npy")
S, w, _ = an.compute_H(gen, nps, np.hanning, sampling_rate=10000)
p = np.abs(S)**2/nps; M0, M2, M4 = p.sum(), (w**2*p).sum(), (w**4*p).sum()
g = S*(M2/M0 - w**2); gn = np.sqrt(M4 - M2**2/M0); N = len(S)
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.9), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)

# (a) nel tempo: singolo contro pile-up, stessa energia, allineati al baricentro
dt, r = 8e-4, 0.5
t = (np.arange(N) - 5000) / 10.0                      # ms
s = gen / gen.max()
pu = (1 - r) * s + r * np.roll(s, int(round(dt * 1e4)))
pu = np.roll(pu, -int(round(r * dt * 1e4)))           # baricentro temporale sulla posizione del singolo
k = (t > -4) & (t < 12)
ax[0].plot(t[k], s[k], color=BLUE, lw=2, label="single pulse")
ax[0].plot(t[k], pu[k], color=ORANGE, lw=2, ls="--", label=f"pile-up, Δt = {dt*1e3:.1f} ms, r = {r}")
ax[0].set_xlabel("time [ms]"); ax[0].set_ylabel("amplitude (single = 1)")
ax[0].set_title("(a) pile-up = a slightly WIDER single pulse", loc="left", color=INK)
ax[0].legend(frameon=False, fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 0.8))

# (b) in frequenza: contributo di ogni banda a T. Asse log -> si disegna peso * f, cosi' le AREE
# sul grafico sono proporzionali ai contributi (sum df = int f dln f) e la cancellazione si vede.
f = w / (2*np.pi); h = f > 0; o = np.argsort(f[h]); fh = f[h][o]
wt = ((M2/M0 - w**2) * p)[h][o] * fh
# fattore del pile-up per uno stimatore LINEARE letto al baricentro: Re[(1-r) e^{i w r dt} + r e^{-i w (1-r) dt}]
D = ((1 - r)*np.cos(w*r*dt) + r*np.cos(w*(1 - r)*dt))[h][o]
sc = np.abs(wt).max()
Tpu = A * ((M2/M0 - w**2) * p * ((1 - r)*np.cos(w*r*dt) + r*np.cos(w*(1 - r)*dt))).sum() / gn
f0 = np.sqrt(M2/M0) / (2*np.pi)
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].fill_between(fh, wt/sc, 0, where=wt > 0, color=BLUE, alpha=0.22, lw=0)
ax[1].fill_between(fh, wt/sc, 0, where=wt < 0, color=ORANGE, alpha=0.22, lw=0)
ax[1].plot(fh, wt/sc, color=INK, lw=1.4, label="single pulse: blue and orange areas are equal → T = 0")
ax[1].plot(fh, wt*D/sc, color=ORANGE, lw=1.6, ls="--",
           label=f"pile-up (Δt = {dt*1e3:.1f} ms, r = {r}): high frequencies damped → T = +{Tpu:.1f} σ")
ax[1].axvline(f0, color=INK2, lw=0.8, ls=":")
ax[1].text(f0*1.07, 0.06, f"f₀ = {f0:.0f} Hz", color=INK2, fontsize=9)
ax[1].text(20, 0.45, "low frequencies\ncount +", color=BLUE, fontsize=9.5, ha="center")
ax[1].text(1500, -0.97, "high frequencies\ncount −", color=ORANGE, fontsize=9.5, ha="center")
ax[1].set_xscale("log"); ax[1].set_xlim(3, 2500); ax[1].set_ylim(-1.15, 1.5); ax[1].set_yticks([-1, -0.5, 0, 0.5, 1])
ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("contribution to T per log-frequency (normalized)")
ax[1].set_title("(b) T = low-frequency minus high-frequency content", loc="left", color=INK)
ax[1].legend(frameon=False, fontsize=8.6, loc="upper right", bbox_to_anchor=(1.02, 1.0))

# (c) la distribuzione di T sul Monte Carlo e il taglio
def T_of(x):
    X = np.fft.fft(x, axis=1); y = np.fft.ifft(X*np.conj(S)/nps, axis=1).real
    idx = np.r_[N-20:N, 0:21]; k = idx[y[:, idx].argmax(axis=1)]; n = np.arange(len(x))
    ym, y0, yp = y[n, (k-1) % N], y[n, k], y[n, (k+1) % N]
    tau = ((k + N//2) % N - N//2 + 0.5*(ym-yp)/(ym-2*y0+yp)) / 1e4
    return (X*np.exp(1j*w[None, :]*tau[:, None])*np.conj(g)[None, :]/nps).real.sum(axis=1)/gn
Ts, Tp = [], []
for L, dtm in ((Ts, 0.0), (Tp, 8e-4)):
    rng = np.random.default_rng(4242)
    for _ in range(30):
        fp, *_ = sim.simulate_frequency_pulses(S, nps, 0.0, w, nsim=500, seed=rng, signal_scale=A, dt_max=dtm)
        L.append(T_of(np.fft.ifft(fp, axis=1).real))
Ts, Tp = np.concatenate(Ts), np.concatenate(Tp); cut = np.percentile(Ts, 90)
bins = np.linspace(-4, 12, 100)
ax[2].hist(Ts, bins, color=BLUE, alpha=0.55, label=f"single pulses: mean {Ts.mean():.2f}, sd {Ts.std():.2f}")
ax[2].hist(Tp, bins, color=ORANGE, alpha=0.55, label=f"pile-up (Δt < 0.8 ms): {100*np.mean(Tp < cut):.0f}% survive")
ax[2].axvline(cut, color=INK, lw=1.6)
top = ax[2].get_ylim()[1]
ax[2].text(cut + 0.3, top*0.80, f"cut T = {cut:.2f}\n(keeps 90 % of singles)", fontsize=9, color=INK, va="top")
ax[2].text(-3.8, top*0.30, "← accepted", ha="left", fontsize=9.5, color=INK2)
ax[2].text(cut + 0.3, top*0.56, "rejected →", ha="left", fontsize=9.5, color=INK2)
ax[2].set_xlim(-4, 12)
ax[2].set_xlabel("width statistic T  [noise σ]"); ax[2].set_ylabel("events")
ax[2].set_title("(c) one number per event, one cut", loc="left", color=INK)
ax[2].legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.02, 1.0))
fig.suptitle(f"Width estimator — m205 Ch{CH} WP{WP}, Monte Carlo 15 000 events per population",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{BASE}/width_estimator_explained.png", dpi=140, facecolor=SURF)
print("taglio", cut, "sopravvivono", np.mean(Tp < cut))

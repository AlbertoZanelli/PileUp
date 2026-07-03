import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# -------------------------
# Your function
# -------------------------
def pole_zero_pulse_components(t, t0, zero, poles):
    diff = poles[:, None] - poles[None, :]
    denom = np.prod(np.where(np.eye(len(poles), dtype=bool), 1.0, diff), axis=1)
    k = (poles - zero) / denom
    B = k
    exp_p = poles
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    components_full = np.zeros((len(poles), len(t)))
    mask = t >= t0
    tt = t[mask] - t0
    components = B[:, None] * np.exp(exp_p[:, None] * tt)
    components_full[:, mask] = components
    out[mask] = np.sum(components, axis=0)
    return components_full/np.max(out), out/np.max(out), k

# -------------------------
# Initial parameters
# -------------------------
t = np.arange(-256 // 2, 512 // 2,0.1) * 1e-4
fit_parameters = np.loadtxt(f"../outputs/fits/pulse_fit_params.csv", delimiter = ',', skiprows = 1)
channel = 19
bessel_order = 0

# -------------------------
# Parameters (EDIT THESE)
# -------------------------
popt = fit_parameters[(fit_parameters[:, 0] == channel) & (fit_parameters[:, 1] == bessel_order), 2:][0]

t0 = popt[0]
τz = 1/(-popt[1])  # zero
τ1 = 1/(-popt[2])
τ2 = 1/(-popt[3])
τ3 = 1/(-popt[4])
τ4 = 1/(-popt[5])

zero = -1 / τz
poles = -1 / np.array([τ1, τ2, τ3, τ4])
poles = np.sort(poles)
print(poles)
# poles = poles[2:]
# -------------------------
# Initial plot
# -------------------------
components, total, k = pole_zero_pulse_components(t, t0, zero, poles)

fig, ax = plt.subplots(figsize = (10, 6))
plt.subplots_adjust(left = 0.1, bottom = 0.35)

lines = []
# components and total already have same length as t
line, = ax.plot(t, components[0]+components[1], linestyle="--", label=f"Pole 1+2")
lines.append(line)
for i, comp in enumerate(components[2:]):
    line, = ax.plot(t, comp, linestyle="--", label=f"Pole {i+3}")
    lines.append(line)

total_line, = ax.plot(t, total, 'k', linewidth=2.5, label="Total pulse")


ax.set_ylim(-3, 3)
ax.legend()

# -------------------------
# Sliders
# -------------------------
axcolor = 'lightgoldenrodyellow'
ax_zero = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor = axcolor)
ax_p1 = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor = axcolor)
ax_p2 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor = axcolor)
ax_p3 = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor = axcolor)
ax_p4 = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor = axcolor)

s_zero = Slider(ax_zero, 'Zero', -10000, -1.0, valinit = zero)
s_p1 = Slider(ax_p1, 'Pole1', -40000, -1.0, valinit = poles[0])
s_p2 = Slider(ax_p2, 'Pole2', -40000, -1.0, valinit = poles[1])
s_p3 = Slider(ax_p3, 'Pole3', -10000, -1.0, valinit = poles[2])
s_p4 = Slider(ax_p4, 'Pole4', -10000, -1.0, valinit = poles[3])


# -------------------------
# Update function
# -------------------------
def update(val):
    new_zero = s_zero.val
    new_poles = np.array([s_p1.val, s_p2.val, s_p3.val, s_p4.val])
    new_components, new_total, _ = pole_zero_pulse_components(t, t0, new_zero, new_poles)
    new_components = new_components[0]+new_components[1], new_components[2], new_components[3]
    for line, comp in zip(lines, new_components):
        line.set_ydata(comp)
    total_line.set_ydata(new_total)
    fig.canvas.draw_idle()


s_zero.on_changed(update)
s_p1.on_changed(update)
s_p2.on_changed(update)
s_p3.on_changed(update)
s_p4.on_changed(update)

plt.show()


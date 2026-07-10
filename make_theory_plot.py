#!/usr/bin/env python3
"""Vẽ hình đối chiếu công thức giải tích với mô phỏng Monte Carlo.

Xuất: outputs/analytic_vs_mc.png
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import build_airspace, SimParams, run_montecarlo
from drone_sim import PHASE_LABELS_VI

OUT = os.path.join(os.path.dirname(__file__), "outputs")
T = 1800.0
DENS = [10, 20, 30, 40, 50, 60]

air = build_airspace(np.random.default_rng(7), num_routes=10, num_types=10)
R = air.num_routes

# ---- Hằng số kỳ vọng lấy trực tiếp từ mô hình ----
mu_to = mu_la = 0.5 * (15 + 40)
mu_d = 0.5 * (0.3 + 1.0) * (0.5 * (120 + 240))
inv_P_to = np.mean([1.0 / len(r.takeoff_point.pads) for r in air.routes])
inv_P_la = np.mean([1.0 / len(r.landing_point.pads) for r in air.routes])
radii = [d.safety_radius for d in air.drone_types]
s_bar = np.mean([max(a, b) for a in radii for b in radii])
v_bar = np.mean([0.5 * (d.speed_min + d.speed_max) for d in air.drone_types])

c_en = 3.5
q = {
    "takeoff": (1.0 / R) * (2 * mu_to / T),
    "landing": (1.0 / R) * (2 * mu_la / T),
    "parking": (inv_P_to / R) * (2 * mu_d / T) + (inv_P_la / R) * (2 * mu_d / T),
    "enroute": c_en * (1.0 / R) * (2 * s_bar / (v_bar * T)),
}


def P_formula(qp, N):
    return 1 - math.exp(-(N - 1) * qp)


# ---- Mô phỏng ----
phases = ["takeoff", "landing", "enroute", "parking"]
mc = {p: [] for p in phases + ["any"]}
for N in DENS:
    res = run_montecarlo(air, SimParams(num_flights=N, horizon=T),
                         trials=2000, seed=123)
    for p in phases + ["any"]:
        mc[p].append(res.phases[p].p_per_flight * 100)

# ---- Vẽ ----
colors = {"takeoff": "#4C72B0", "landing": "#55A868",
          "enroute": "#C44E52", "parking": "#8172B3", "any": "#333333"}
Ngrid = np.linspace(DENS[0], DENS[-1], 100)

fig, ax = plt.subplots(figsize=(9.5, 6))
for p in phases:
    ax.plot(Ngrid, [P_formula(q[p], n) * 100 for n in Ngrid], "-",
            color=colors[p], lw=2, alpha=0.9,
            label=f"{PHASE_LABELS_VI[p]} — công thức")
    ax.plot(DENS, mc[p], "o", color=colors[p], ms=7, mec="black", mew=0.6,
            label=f"{PHASE_LABELS_VI[p]} — Monte Carlo")

# 'any': cận trên độc lập (đứt) + MC
indep = [(1 - np.prod([1 - P_formula(q[p], n) for p in phases])) * 100 for n in Ngrid]
ax.plot(Ngrid, indep, "--", color=colors["any"], lw=1.6,
        label="Toàn KHB — cận trên (độc lập)")
ax.plot(DENS, mc["any"], "s", color=colors["any"], ms=7, mec="black", mew=0.6,
        label="Toàn KHB — Monte Carlo")

ax.set_xlabel("Số kế hoạch bay N (mật độ giao thông)")
ax.set_ylabel("Xác suất xung đột mỗi KHB (%)")
ax.set_title("Công thức  $P_p = 1-e^{-(N-1)a_p/T}$  vs  Monte Carlo\n"
             f"(T = {T/60:.0f} phút, R = {R} đường bay, 2000 trial/điểm)")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8, framealpha=0.9)
fig.tight_layout()
path = os.path.join(OUT, "analytic_vs_mc.png")
fig.savefig(path, dpi=120)
print("đã lưu:", path)
print("a_p (giây):", {k: round(v * T, 2) for k, v in q.items()})

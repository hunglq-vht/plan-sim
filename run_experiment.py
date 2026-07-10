#!/usr/bin/env python3
"""Thí nghiệm Monte Carlo: xác suất xung đột đường bay drone.

Cách dùng
---------
    python run_experiment.py                 # chạy với tham số mặc định
    python run_experiment.py --trials 5000 --flights 40
    python run_experiment.py --no-plots      # bỏ qua vẽ hình
    python run_experiment.py --sweep 10 20 30 40 50 60   # quét mật độ

Kết quả in ra bảng xác suất theo từng pha và (mặc định) lưu các hình vào
thư mục ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

from drone_sim import (
    build_airspace,
    SimParams,
    run_montecarlo,
    PHASE_LABELS_VI,
)
from drone_sim.montecarlo import PHASES

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# --------------------------------------------------------------------------- #
def describe_airspace(airspace) -> str:
    lines = []
    lines.append(f"Vùng trời: {airspace.size:.0f} m x {airspace.size:.0f} m, "
                 f"{airspace.num_routes} đường bay, "
                 f"khoảng cách an toàn tối thiểu {airspace.min_safety:.0f} m")
    lines.append("")
    lines.append("Loại drone (bán kính an toàn / dải vận tốc):")
    for d in airspace.drone_types:
        lines.append(f"  {d.name}: an toàn {d.safety_radius:5.1f} m, "
                     f"vận tốc {d.speed_min:4.1f}–{d.speed_max:4.1f} m/s")
    lines.append("")
    lines.append("Đường bay:")
    for r in airspace.routes:
        n_to = len(r.takeoff_point.pads)
        n_la = len(r.landing_point.pads)
        padt = [f"{p.max_dwell/60:.1f}" for p in r.takeoff_point.pads]
        lines.append(
            f"  R{r.id:02d}: {r.num_waypoints} waypoint, độ cao {r.altitude:5.1f} m, "
            f"cất cánh {n_to} bãi đỗ, hạ cánh {n_la} bãi đỗ, "
            f"pad_t(cất) = [{', '.join(padt)}] phút"
        )
    return "\n".join(lines)


def print_result_table(result) -> None:
    print(f"\nKết quả Monte Carlo — {result.trials} trial, "
          f"{result.params.num_flights} kế hoạch bay/trial, "
          f"horizon {result.params.horizon/60:.0f} phút")
    print(f"Tổng số kế hoạch bay mô phỏng: {result.phases['any'].flights_total:,}")
    print("-" * 78)
    print(f"{'Pha':<20}{'P(theo KHB)':>14}{'  KTC 95%':>20}{'P(theo kịch bản)':>20}")
    print("-" * 78)
    for pr in result.summary_rows():
        lo, hi = pr.ci95_per_flight()
        label = PHASE_LABELS_VI[pr.phase]
        marker = "  <== toàn bộ" if pr.phase == "any" else ""
        print(f"{label:<20}{pr.p_per_flight*100:>12.3f}%"
              f"   [{lo*100:5.3f}%, {hi*100:5.3f}%]"
              f"{pr.p_per_scenario*100:>17.2f}%{marker}")
    print("-" * 78)
    print("P(theo KHB)      : xác suất một kế hoạch bay dính xung đột ở pha đó")
    print("P(theo kịch bản) : xác suất một trial có >= 1 xung đột ở pha đó")


# --------------------------------------------------------------------------- #
def plot_airspace(airspace, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for r in airspace.routes:
        color = cmap(r.id % 10)
        nodes = r.path_nodes()
        ax.plot(nodes[:, 0], nodes[:, 1], "-", color=color, lw=1.5, alpha=0.8,
                label=f"R{r.id}")
        # waypoint
        ax.plot(r.waypoints[:, 0], r.waypoints[:, 1], "o", color=color, ms=4)
        # điểm cất cánh (tam giác) và hạ cánh (vuông)
        tp, lp = r.takeoff_point.pos, r.landing_point.pos
        ax.plot(tp[0], tp[1], "^", color=color, ms=11, mec="black")
        ax.plot(lp[0], lp[1], "s", color=color, ms=11, mec="black")
        # bãi đỗ (chấm nhỏ)
        for op in (r.takeoff_point, r.landing_point):
            pads = np.array([p.pos for p in op.pads])
            ax.plot(pads[:, 0], pads[:, 1], ".", color=color, ms=3, alpha=0.6)

    ax.set_xlim(0, airspace.size)
    ax.set_ylim(0, airspace.size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vùng trời: 10 đường bay không giao nhau\n"
                 "▲ cất cánh   ■ hạ cánh   ● waypoint")
    ax.legend(loc="upper center", ncol=5, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")


def plot_phase_bars(result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(PHASES) + ["any"]
    labels = [PHASE_LABELS_VI[p] for p in order]
    probs = [result.phases[p].p_per_flight * 100 for p in order]
    cis = [result.phases[p].ci95_per_flight() for p in order]
    err = [[probs[i] - cis[i][0] * 100 for i in range(len(order))],
           [cis[i][1] * 100 - probs[i] for i in range(len(order))]]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#333333"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, probs, color=colors, yerr=err, capsize=4)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Xác suất xung đột theo mỗi kế hoạch bay (%)")
    ax.set_title(f"Xác suất xung đột theo pha\n"
                 f"({result.trials} trial × {result.params.num_flights} KHB, "
                 f"khoảng tin cậy 95%)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")


def plot_density_sweep(airspace, base_params, densities, trials, seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(PHASES) + ["any"]
    series = {p: [] for p in order}
    print("\nQuét mật độ giao thông:")
    for n in densities:
        params = SimParams(**{**base_params.__dict__, "num_flights": n})
        res = run_montecarlo(airspace, params, trials=trials, seed=seed)
        for p in order:
            series[p].append(res.phases[p].p_per_flight * 100)
        print(f"  N={n:3d}: toàn bộ = {res.phases['any'].p_per_flight*100:6.2f}%")

    colors = {"takeoff": "#4C72B0", "landing": "#55A868",
              "enroute": "#C44E52", "parking": "#8172B3", "any": "#333333"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for p in order:
        ax.plot(densities, series[p], "o-", color=colors[p],
                lw=2 if p == "any" else 1.5,
                label=PHASE_LABELS_VI[p])
    ax.set_xlabel("Số kế hoạch bay trên mỗi trial (mật độ giao thông)")
    ax.set_ylabel("Xác suất xung đột theo mỗi KHB (%)")
    ax.set_title("Xác suất xung đột theo mật độ giao thông")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")
    return {p: series[p] for p in order}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=3000, help="số trial Monte Carlo")
    ap.add_argument("--flights", type=int, default=30, help="số KHB mỗi trial")
    ap.add_argument("--horizon", type=float, default=1800.0, help="cửa sổ thời gian (s)")
    ap.add_argument("--routes", type=int, default=10)
    ap.add_argument("--types", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--airspace-seed", type=int, default=7)
    ap.add_argument("--dt", type=float, default=1.0, help="bước lấy mẫu pha hành trình (s)")
    ap.add_argument("--sweep", type=int, nargs="*",
                    default=[10, 20, 30, 40, 50, 60],
                    help="danh sách mật độ để quét (rỗng để bỏ qua)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Vùng trời tĩnh (seed riêng để tái lập cấu hình)
    rng_air = np.random.default_rng(args.airspace_seed)
    airspace = build_airspace(rng_air, num_routes=args.routes, num_types=args.types)
    print(describe_airspace(airspace))

    # 2) Monte Carlo ở mật độ mặc định
    params = SimParams(num_flights=args.flights, horizon=args.horizon,
                       enroute_dt=args.dt)

    def progress(t, n):
        print(f"\r  Monte Carlo: {t}/{n} trial", end="", flush=True)

    result = run_montecarlo(airspace, params, trials=args.trials,
                            seed=args.seed, progress=progress)
    print()
    print_result_table(result)

    # 3) Lưu kết quả JSON
    result_json = {
        "config": {
            "trials": args.trials, "flights": args.flights,
            "horizon": args.horizon, "routes": args.routes,
            "types": args.types, "seed": args.seed,
            "airspace_seed": args.airspace_seed, "dt": args.dt,
        },
        "phases": {
            pr.phase: {
                "p_per_flight": pr.p_per_flight,
                "ci95_per_flight": pr.ci95_per_flight(),
                "p_per_scenario": pr.p_per_scenario,
                "flights_total": pr.flights_total,
                "flights_conflicted": pr.flights_conflicted,
            }
            for pr in result.summary_rows()
        },
    }

    # 4) Hình vẽ
    if not args.no_plots:
        print("\nVẽ hình:")
        plot_airspace(airspace, os.path.join(OUT_DIR, "airspace_map.png"))
        plot_phase_bars(result, os.path.join(OUT_DIR, "phase_probabilities.png"))
        if args.sweep:
            sweep = plot_density_sweep(
                airspace, params, args.sweep,
                trials=max(1000, args.trials // 3),
                seed=args.seed,
                path=os.path.join(OUT_DIR, "density_sweep.png"),
            )
            result_json["sweep"] = {"densities": args.sweep, "series": sweep}

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu kết quả: {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()

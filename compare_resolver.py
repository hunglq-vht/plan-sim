#!/usr/bin/env python3
"""Thí nghiệm đối chứng: baseline (chưa điều phối) vs thuật toán time-departure.

Ghép bộ giải xung đột ``scrp_simple.ConflictResolver`` (repo
hunglq-vht/time-departure) vào bộ Monte Carlo của plan-sim:

  1. Sinh N kế hoạch bay y hệt baseline.
  2. BASELINE : đo xung đột trực tiếp (không điều phối).
  3. RESOLVED : cho ``resolve_batch`` duyệt tuần tự — mỗi yêu cầu có thể bị
     dời giờ xuất phát (delay), được gán bãi đỗ hạ cánh, hoặc bị TỪ CHỐI nếu
     delay > max_wait. Sau đó đo lại xung đột trên tập đã duyệt bằng ĐÚNG bộ
     phát hiện của baseline để so sánh công bằng.

Ánh xạ mô hình
--------------
  route.takeoff_point  -> Vertiport (pad = các bãi đỗ cất cánh)
  route.landing_point  -> Vertiport (pad = các bãi đỗ hạ cánh)
  route.waypoints      -> FlightPath.waypoints (chỉ waypoint trung gian)
  pad.max_dwell        -> Pad.occupation_duration
  C1<->hành trình, C2<->cất cánh, C3<->hạ cánh, C4<->bãi đỗ (hạ cánh)

Lưu ý: scrp_simple quản lý bãi đỗ HẠ CÁNH (C4) + tuần tự hoá vùng trời, nhưng
KHÔNG quản lý bãi đỗ CẤT CÁNH. Vì vậy ta tách pha "Bãi đỗ" thành hai kênh
(hạ cánh / cất cánh) để thấy rõ phần dư thuộc về đâu.

Cách dùng
---------
    python compare_resolver.py                       # mặc định
    python compare_resolver.py --resolver-path /home/user/time-departure
    python compare_resolver.py --sweep 10 20 30 40 50 60 --max-wait 600
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from drone_sim import build_airspace, SimParams, generate_flight_plans
from drone_sim import conflicts as C
from drone_sim.flight import FlightPlan
from drone_sim.montecarlo import _wilson_ci

OUT = os.path.join(os.path.dirname(__file__), "outputs")


# --------------------------------------------------------------------------- #
# Nạp resolver từ repo time-departure
# --------------------------------------------------------------------------- #
def load_resolver(path: str):
    if not os.path.isdir(os.path.join(path, "scrp_simple")):
        sys.exit(f"Không thấy scrp_simple trong {path!r}. "
                 f"Hãy checkout repo hunglq-vht/time-departure và trỏ --resolver-path.")
    sys.path.insert(0, path)
    from scrp_simple import models as M
    from scrp_simple.resolver import resolve_batch
    return M, resolve_batch


# --------------------------------------------------------------------------- #
# Chuyển airspace/flight của plan-sim sang mô hình của resolver
# --------------------------------------------------------------------------- #
def build_vertiports(airspace, M):
    """Mỗi điểm cất/hạ cánh -> một Vertiport; bãi đỗ -> Pad."""
    verts = {}
    for r in airspace.routes:
        for op in (r.takeoff_point, r.landing_point):
            pads = [M.Pad(id=str(p.id), occupation_duration=p.max_dwell)
                    for p in op.pads]
            verts[str(op.id)] = M.Vertiport(
                id=str(op.id),
                location=M.Point3D(*op.pos.tolist()),
                pads=pads,
            )
    return verts


def flight_to_request(f: FlightPlan, M, max_wait: float):
    wps = [M.Point3D(*w.tolist()) for w in f.route.waypoints]
    fp = M.FlightPath(
        takeoff_vertiport_id=str(f.route.takeoff_point.id),
        landing_vertiport_id=str(f.route.landing_point.id),
        waypoints=wps,
    )
    return M.NewPlanRequest(
        flight_path=fp,
        drone_type=f.drone_type.name,
        segment_speeds=f.seg_speeds.tolist(),
        desired_start_time=f.departure_time,
        max_wait_time=max_wait,
        t_takeoff=f.takeoff_time,
        t_land=f.landing_time,
        priority=5,
    )


def apply_resolution(airspace, flights, M, resolve_batch, max_wait):
    """Chạy resolver; trả về (tập KHB đã duyệt (đã cập nhật giờ + bãi đỗ), stats)."""
    verts = build_vertiports(airspace, M)
    requests = [flight_to_request(f, M, max_wait) for f in flights]
    results = resolve_batch(verts, [], requests)

    resolved: List[FlightPlan] = []
    delays: List[float] = []
    n_rejected = 0
    for f, res in zip(flights, results):
        if not res.approved:
            n_rejected += 1
            continue
        delays.append(res.delay)
        # tạo bản sao KHB với giờ xuất phát mới + bãi đỗ hạ cánh do resolver gán
        g = copy.copy(f)
        g.departure_time = res.actual_start_time
        if res.assigned_pad_id is not None:
            for p in f.route.landing_point.pads:
                if str(p.id) == res.assigned_pad_id:
                    g.landing_pad = p
                    break
        g.__post_init__()               # tính lại quỹ đạo/mốc thời gian
        resolved.append(g)
    stats = {
        "n_total": len(flights),
        "n_approved": len(resolved),
        "n_rejected": n_rejected,
        "delays": delays,
    }
    return resolved, stats


# --------------------------------------------------------------------------- #
# Phát hiện xung đột (tách bãi đỗ thành 2 kênh cất/hạ cánh)
# --------------------------------------------------------------------------- #
# Dung sai bỏ qua tiếp xúc đúng biên (resolver xếp cửa sổ sát nhau);
# áp dụng ĐỒNG NHẤT cho cả baseline lẫn resolved để so sánh công bằng.
EPS_T = 1e-3   # giây
EPS_M = 1e-2   # mét


def _pad_window(f, which: str):
    """Cửa sổ chiếm bãi đỗ, dùng occupation_duration = pad.max_dwell.

    Nhất quán với mô hình của resolver (bãi bận đúng ``occupation_duration``
    sau chạm đất / trước khi cất cánh), tránh phụ thuộc dwell ngẫu nhiên gắn
    với bãi đỗ gốc trước khi bị gán lại.
    """
    if which == "takeoff":
        pad = f.takeoff_pad
        return pad.id, (f.departure_time - pad.max_dwell, f.departure_time)
    pad = f.landing_pad
    return pad.id, (f.arrival_time, f.arrival_time + pad.max_dwell)


def _landing_pad_conflicts(flights):
    """Xung đột bề mặt pad HẠ CÁNH (đúng ràng buộc C4 của scrp_simple).

    scrp_simple chỉ ràng buộc bề mặt pad khi hạ cánh (pad bận
    ``occupation_duration`` sau chạm đất). Cất cánh chỉ chiếm KHÔNG PHẬN (C2),
    không phải bề mặt pad — nên KHÔNG có kênh 'bãi đỗ cất cánh' riêng.
    """
    groups: Dict[int, List[Tuple[float, float, int]]] = {}
    for f in flights:
        pad_id, w = _pad_window(f, "landing")
        groups.setdefault(pad_id, []).append((w[0], w[1], f.id))
    conflicted = set()
    for items in groups.values():
        if len(items) < 2:
            continue
        for i, j in C._sweep_overlaps(items, eps=EPS_T):
            conflicted.add(i)
            conflicted.add(j)
    return conflicted


def detect_phases(flights, airspace, dt=1.0):
    """Xung đột theo đúng bốn ràng buộc của scrp_simple (C1..C4).

    Khoảng cách an toàn dùng ``airspace.pairwise_safety`` — để khớp mô hình
    scrp_simple (MIN_SEPARATION = 50m cố định), gọi ``unify_safety`` trước.
    """
    to_c, _ = C.detect_takeoff_conflicts(flights, eps=EPS_T)         # C2
    la_c, _ = C.detect_landing_conflicts(flights, eps=EPS_T)         # C3
    en_c, _ = C.detect_enroute_conflicts(flights, airspace, dt=dt, eps_m=EPS_M)  # C1
    pk_c = _landing_pad_conflicts(flights)                           # C4
    out = {"takeoff": to_c, "landing": la_c, "enroute": en_c, "parking": pk_c}
    out["any"] = to_c | la_c | en_c | pk_c
    return out


def unify_safety(airspace):
    """Đặt bán kính an toàn của MỌI loại drone = min_safety (50m).

    scrp_simple không có MSD theo loại (drone_type chỉ là chuỗi) và enforce
    MIN_SEPARATION_M = 50 cố định; đối chứng vì vậy dùng 50m thống nhất.
    """
    for d in airspace.drone_types:
        d.safety_radius = airspace.min_safety


PHASES = ["takeoff", "landing", "enroute", "parking", "any"]
LABELS = {
    "takeoff": "Cất cánh (C2)", "landing": "Hạ cánh (C3)",
    "enroute": "Trên hành trình (C1)", "parking": "Bãi đỗ hạ cánh (C4)",
    "any": "Toàn kế hoạch bay",
}


# --------------------------------------------------------------------------- #
# Chạy đối chứng ở một mật độ
# --------------------------------------------------------------------------- #
def run_compare(airspace, M, resolve_batch, N, horizon, max_wait,
                trials, seed, dt=1.0):
    rng = np.random.default_rng(seed)
    params = SimParams(num_flights=N, horizon=horizon, enroute_dt=dt)

    base_conf = {p: 0 for p in PHASES}
    res_conf = {p: 0 for p in PHASES}
    base_total = 0
    res_total = 0
    tot_rejected = 0
    all_delays: List[float] = []

    for _ in range(trials):
        flights = generate_flight_plans(airspace, rng, params)
        base = detect_phases(flights, airspace, dt=dt)
        base_total += len(flights)
        for p in PHASES:
            base_conf[p] += len(base[p])

        resolved, stats = apply_resolution(airspace, flights, M,
                                            resolve_batch, max_wait)
        tot_rejected += stats["n_rejected"]
        all_delays += stats["delays"]
        res = detect_phases(resolved, airspace, dt=dt)
        res_total += len(resolved)
        for p in PHASES:
            res_conf[p] += len(res[p])

    def frac(k, n):
        return k / n if n else 0.0

    delays = np.array(all_delays) if all_delays else np.array([0.0])
    return {
        "N": N,
        "base_total": base_total,
        "res_total": res_total,
        "rejected": tot_rejected,
        "rejection_rate": frac(tot_rejected, base_total),
        "delay_mean": float(delays.mean()),
        "delay_p95": float(np.percentile(delays, 95)),
        "delay_max": float(delays.max()),
        "baseline": {p: frac(base_conf[p], base_total) for p in PHASES},
        "resolved": {p: frac(res_conf[p], res_total) for p in PHASES},
        "baseline_ci": {p: _wilson_ci(base_conf[p], base_total) for p in PHASES},
        "resolved_ci": {p: _wilson_ci(res_conf[p], res_total) for p in PHASES},
    }


# --------------------------------------------------------------------------- #
def print_table(r):
    print(f"\n{'='*72}")
    print(f"Đối chứng tại N = {r['N']} KHB  "
          f"(baseline {r['base_total']:,} chuyến / resolved {r['res_total']:,} chuyến)")
    print(f"Từ chối: {r['rejected']:,} ({r['rejection_rate']*100:.2f}%)   "
          f"Trễ TB {r['delay_mean']:.1f}s  p95 {r['delay_p95']:.1f}s  "
          f"max {r['delay_max']:.1f}s")
    print("-" * 72)
    print(f"{'Pha':<22}{'Baseline':>12}{'Resolved':>12}{'Giảm':>12}")
    print("-" * 72)
    for p in PHASES:
        b = r["baseline"][p] * 100
        s = r["resolved"][p] * 100
        red = "—" if b == 0 else f"{(1 - s/b)*100:5.1f}%"
        print(f"{LABELS[p]:<22}{b:>11.3f}%{s:>11.3f}%{red:>12}")
    print("-" * 72)


def plot_compare(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["takeoff", "landing", "enroute", "parking", "any"]
    labels = [LABELS[p] for p in order]
    base = [r["baseline"][p] * 100 for p in order]
    res = [r["resolved"][p] * 100 for p in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w/2, base, w, label="Baseline (chưa điều phối)",
                color="#C44E52")
    b2 = ax.bar(x + w/2, res, w, label="Resolved (time-departure)",
                color="#55A868")
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width()/2, bb.get_height(),
                    f"{bb.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Xác suất xung đột mỗi KHB (%)")
    ax.set_title(f"Baseline vs thuật toán time-departure (N={r['N']})\n"
                 f"Từ chối {r['rejection_rate']*100:.1f}%, "
                 f"trễ TB {r['delay_mean']:.0f}s")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolver-path", default="/home/user/time-departure")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--flights", type=int, default=30)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--max-wait", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--airspace-seed", type=int, default=7)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sweep", type=int, nargs="*", default=[10, 20, 30, 40, 50, 60])
    ap.add_argument("--maxwait-sweep", type=float, nargs="*",
                    default=[15, 30, 60, 120, 300, 600],
                    help="quét max_wait (tại mật độ --maxwait-N) để xem đánh đổi từ chối")
    ap.add_argument("--maxwait-N", type=int, default=60,
                    help="mật độ dùng cho quét max_wait")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    M, resolve_batch = load_resolver(args.resolver_path)
    airspace = build_airspace(np.random.default_rng(args.airspace_seed),
                              num_routes=10, num_types=10)
    unify_safety(airspace)   # khớp mô hình scrp_simple: 50m thống nhất

    print(f"So sánh Baseline vs time-departure — {args.trials} trial, "
          f"max_wait {args.max_wait:.0f}s, horizon {args.horizon/60:.0f} phút")
    print("Mô hình khớp scrp_simple: ràng buộc pad = HẠ CÁNH (C4); "
          "khoảng cách an toàn 50m thống nhất")

    main_res = run_compare(airspace, M, resolve_batch, args.flights,
                           args.horizon, args.max_wait, args.trials,
                           args.seed, dt=args.dt)
    print_table(main_res)

    out_json = {"config": vars(args), "main": main_res}

    if args.sweep:
        print("\nQuét mật độ (any / rejection / delay):")
        sweep = []
        for N in args.sweep:
            r = run_compare(airspace, M, resolve_batch, N, args.horizon,
                            args.max_wait, max(400, args.trials // 2),
                            args.seed, dt=args.dt)
            sweep.append(r)
            print(f"  N={N:3d}: baseline any={r['baseline']['any']*100:5.2f}%  "
                  f"resolved any={r['resolved']['any']*100:5.2f}%  "
                  f"từ chối={r['rejection_rate']*100:5.2f}%  "
                  f"trễ TB={r['delay_mean']:5.1f}s")
        out_json["sweep"] = sweep

    if args.maxwait_sweep:
        print(f"\nQuét max_wait tại N={args.maxwait_N} "
              f"(xung đột luôn 0; đổi lấy từ chối/trễ):")
        mw_sweep = []
        for mw in args.maxwait_sweep:
            r = run_compare(airspace, M, resolve_batch, args.maxwait_N,
                            args.horizon, mw, max(400, args.trials // 2),
                            args.seed, dt=args.dt)
            r["max_wait"] = mw
            mw_sweep.append(r)
            print(f"  max_wait={mw:5.0f}s: resolved any={r['resolved']['any']*100:4.2f}%  "
                  f"từ chối={r['rejection_rate']*100:5.2f}%  "
                  f"trễ TB={r['delay_mean']:5.1f}s")
        out_json["maxwait_sweep"] = mw_sweep

    if not args.no_plots:
        print("\nVẽ hình:")
        plot_compare(main_res, os.path.join(OUT, "baseline_vs_resolved.png"))
        if args.sweep:
            plot_sweep(out_json["sweep"], os.path.join(OUT, "resolver_sweep.png"))
        if args.maxwait_sweep:
            plot_maxwait(out_json["maxwait_sweep"], args.maxwait_N,
                         os.path.join(OUT, "resolver_maxwait.png"))

    with open(os.path.join(OUT, "compare_results.json"), "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {os.path.join(OUT, 'compare_results.json')}")


def plot_sweep(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = [r["N"] for r in sweep]
    base_any = [r["baseline"]["any"] * 100 for r in sweep]
    res_any = [r["resolved"]["any"] * 100 for r in sweep]
    rej = [r["rejection_rate"] * 100 for r in sweep]
    delay = [r["delay_mean"] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(N, base_any, "o-", color="#C44E52", label="Baseline: P(xung đột)")
    ax1.plot(N, res_any, "s-", color="#55A868", label="Resolved: P(xung đột)")
    ax1.plot(N, rej, "^--", color="#4C72B0", label="Resolved: tỉ lệ từ chối")
    ax1.set_xlabel("Số kế hoạch bay N (mật độ)")
    ax1.set_ylabel("Phần trăm (%)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(N, delay, "d:", color="#8172B3", label="Resolved: trễ TB (s)")
    ax2.set_ylabel("Độ trễ trung bình (s)", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8)
    ax1.set_title("time-departure: chi phí (trễ, từ chối) đổi lấy an toàn theo mật độ")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")


def plot_maxwait(sweep, N, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mw = [r["max_wait"] for r in sweep]
    rej = [r["rejection_rate"] * 100 for r in sweep]
    res_any = [r["resolved"]["any"] * 100 for r in sweep]
    delay = [r["delay_mean"] for r in sweep]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(mw, rej, "^-", color="#4C72B0", label="Tỉ lệ từ chối")
    ax1.plot(mw, res_any, "s-", color="#55A868",
             label="P(xung đột) sau điều phối")
    ax1.set_xlabel("max_wait — độ trễ tối đa cho phép (s)")
    ax1.set_ylabel("Phần trăm (%)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(mw, delay, "d:", color="#8172B3", label="Trễ TB của chuyến được duyệt (s)")
    ax2.set_ylabel("Độ trễ trung bình (s)", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
    ax1.set_title(f"Đánh đổi từ chối ↔ độ trễ (N={N}); xung đột luôn = 0\n"
                  f"max_wait chặt → nhiều từ chối hơn, nhưng chuyến bị từ chối "
                  f"KHÔNG tính vi phạm")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  đã lưu: {path}")


if __name__ == "__main__":
    main()

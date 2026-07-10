"""Sinh kế hoạch bay ngẫu nhiên và chạy thí nghiệm Monte Carlo.

Quy ước xác suất
----------------
Đại lượng chính là *xác suất theo mỗi kế hoạch bay*: gộp toàn bộ chuyến bay của
tất cả các trial, tỉ lệ chuyến bay dính ít nhất một xung đột ở pha tương ứng.

    P_phase = (số chuyến bay dính xung đột ở pha đó) / (tổng số chuyến bay)

Pha "Toàn kế hoạch bay" (any) = dính xung đột ở BẤT KỲ pha nào.

Ngoài ra còn báo cáo *xác suất theo kịch bản*: tỉ lệ trial có >= 1 xung đột ở
pha đó (hữu ích khi đánh giá cả một đợt vận hành).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import math

import numpy as np

from .airspace import Airspace
from .flight import FlightPlan
from . import conflicts as C


PHASES = ("takeoff", "landing", "enroute", "parking")


@dataclass
class SimParams:
    """Tham số điều khiển việc sinh kế hoạch bay và mật độ giao thông."""

    num_flights: int = 30            # số kế hoạch bay trong một trial
    horizon: float = 1800.0          # giây, cửa sổ thời điểm tạo (30 phút)
    depart_delay: tuple = (30.0, 300.0)   # trễ từ lúc tạo đến lúc xuất phát (s)
    takeoff_time: tuple = (15.0, 40.0)    # thời gian cất cánh (s)
    landing_time: tuple = (15.0, 40.0)    # thời gian hạ cánh (s)
    enroute_dt: float = 1.0          # bước lấy mẫu thời gian pha hành trình (s)


# --------------------------------------------------------------------------- #
# Sinh kế hoạch bay
# --------------------------------------------------------------------------- #
def generate_flight_plans(
    airspace: Airspace,
    rng: np.random.Generator,
    params: SimParams,
) -> List[FlightPlan]:
    """Sinh lần lượt ``params.num_flights`` kế hoạch bay.

    Thời điểm tạo là một quá trình đến rải đều trên ``[0, horizon]`` (sắp tăng).
    Mỗi kế hoạch chọn ngẫu nhiên: đường bay, loại drone, bãi đỗ đi/đến, vận tốc
    từng chặng, thời gian cất/hạ cánh, thời gian dừng bãi đỗ (<= pad_t).
    """
    # thời điểm tạo: sắp xếp tăng dần các mốc ngẫu nhiên đều trên horizon
    creation_times = np.sort(rng.uniform(0.0, params.horizon, params.num_flights))

    flights: List[FlightPlan] = []
    for k in range(params.num_flights):
        route = airspace.routes[int(rng.integers(0, airspace.num_routes))]
        dtype = airspace.drone_types[int(rng.integers(0, len(airspace.drone_types)))]

        creation = float(creation_times[k])
        departure = creation + float(rng.uniform(*params.depart_delay))

        num_cruise = route.num_waypoints - 1
        seg_speeds = np.array(
            [dtype.sample_speed(rng) for _ in range(num_cruise)], dtype=float
        )

        takeoff_pad = route.takeoff_point.pads[
            int(rng.integers(0, len(route.takeoff_point.pads)))
        ]
        landing_pad = route.landing_point.pads[
            int(rng.integers(0, len(route.landing_point.pads)))
        ]
        # thời gian dừng bãi đỗ không quá pad_t của bãi đó
        origin_dwell = float(rng.uniform(0.3, 1.0) * takeoff_pad.max_dwell)
        dest_dwell = float(rng.uniform(0.3, 1.0) * landing_pad.max_dwell)

        flights.append(
            FlightPlan(
                id=k,
                route=route,
                drone_type=dtype,
                creation_time=creation,
                departure_time=departure,
                takeoff_time=float(rng.uniform(*params.takeoff_time)),
                landing_time=float(rng.uniform(*params.landing_time)),
                seg_speeds=seg_speeds,
                takeoff_pad=takeoff_pad,
                landing_pad=landing_pad,
                origin_dwell=origin_dwell,
                dest_dwell=dest_dwell,
            )
        )
    return flights


# --------------------------------------------------------------------------- #
# Một trial
# --------------------------------------------------------------------------- #
def run_trial(
    airspace: Airspace,
    rng: np.random.Generator,
    params: SimParams,
) -> Dict[str, object]:
    """Chạy một trial: sinh kế hoạch bay và phát hiện xung đột.

    Trả về:
        n_flights            : số chuyến bay
        conflicted[phase]    : tập id chuyến bay dính xung đột theo từng pha
        conflicted['any']    : tập id dính xung đột ở bất kỳ pha nào
    """
    flights = generate_flight_plans(airspace, rng, params)
    per_phase = C.detect_all(flights, airspace, dt=params.enroute_dt)

    any_set = set()
    for s in per_phase.values():
        any_set |= s
    per_phase_out = dict(per_phase)
    per_phase_out["any"] = any_set

    return {"n_flights": len(flights), "conflicted": per_phase_out}


# --------------------------------------------------------------------------- #
# Kết quả tổng hợp
# --------------------------------------------------------------------------- #
@dataclass
class PhaseResult:
    """Ước lượng xác suất cho một pha."""

    phase: str
    # theo mỗi kế hoạch bay
    flights_total: int
    flights_conflicted: int
    # theo mỗi kịch bản (trial)
    trials_total: int
    trials_with_conflict: int

    @property
    def p_per_flight(self) -> float:
        return self.flights_conflicted / self.flights_total if self.flights_total else 0.0

    @property
    def p_per_scenario(self) -> float:
        return self.trials_with_conflict / self.trials_total if self.trials_total else 0.0

    def ci95_per_flight(self) -> tuple:
        return _wilson_ci(self.flights_conflicted, self.flights_total)

    def ci95_per_scenario(self) -> tuple:
        return _wilson_ci(self.trials_with_conflict, self.trials_total)


@dataclass
class MonteCarloResult:
    params: SimParams
    trials: int
    phases: Dict[str, PhaseResult] = field(default_factory=dict)

    def summary_rows(self):
        order = list(PHASES) + ["any"]
        for ph in order:
            yield self.phases[ph]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Khoảng tin cậy Wilson 95% cho tỉ lệ (ổn định khi p gần 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Chạy Monte Carlo
# --------------------------------------------------------------------------- #
def run_montecarlo(
    airspace: Airspace,
    params: SimParams,
    trials: int,
    seed: int = 12345,
    progress: Optional[callable] = None,
) -> MonteCarloResult:
    """Chạy ``trials`` trial độc lập, tổng hợp xác suất theo từng pha."""
    rng = np.random.default_rng(seed)

    all_phases = list(PHASES) + ["any"]
    flights_total = 0
    flights_conf = {ph: 0 for ph in all_phases}
    trials_conf = {ph: 0 for ph in all_phases}

    for t in range(trials):
        out = run_trial(airspace, rng, params)
        flights_total += out["n_flights"]
        for ph in all_phases:
            c = len(out["conflicted"][ph])
            flights_conf[ph] += c
            if c > 0:
                trials_conf[ph] += 1
        if progress and (t + 1) % max(1, trials // 20) == 0:
            progress(t + 1, trials)

    phases = {}
    for ph in all_phases:
        phases[ph] = PhaseResult(
            phase=ph,
            flights_total=flights_total,
            flights_conflicted=flights_conf[ph],
            trials_total=trials,
            trials_with_conflict=trials_conf[ph],
        )
    return MonteCarloResult(params=params, trials=trials, phases=phases)

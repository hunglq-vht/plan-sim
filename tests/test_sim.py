"""Kiểm thử bất biến của mô phỏng.

Chạy:  python -m pytest tests/ -q      (nếu có pytest)
   hoặc: python tests/test_sim.py       (chạy trực tiếp, không cần pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_sim import build_airspace, SimParams, generate_flight_plans
from drone_sim import conflicts as C


def _segment_intersect(p1, p2, p3, p4) -> bool:
    """Hai đoạn thẳng 2D (p1p2, p3p4) có cắt nhau thực sự không?"""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def test_routes_do_not_cross():
    """Không có hai đường bay nào cắt nhau (chiếu xuống mặt phẳng x-y)."""
    rng = np.random.default_rng(7)
    air = build_airspace(rng, num_routes=10)
    assert air.num_routes == 10
    polylines = [r.path_nodes()[:, :2] for r in air.routes]
    for a in range(len(polylines)):
        for b in range(a + 1, len(polylines)):
            pa, pb = polylines[a], polylines[b]
            for i in range(len(pa) - 1):
                for j in range(len(pb) - 1):
                    assert not _segment_intersect(pa[i], pa[i + 1],
                                                  pb[j], pb[j + 1]), \
                        f"Đường bay {a} và {b} cắt nhau"


def test_airspace_spec():
    """Kiểm tra đúng đặc tả: waypoint 2..4, bãi đỗ 3..5, pad_t 2..4 phút, an toàn>=50."""
    rng = np.random.default_rng(7)
    air = build_airspace(rng, num_routes=10, num_types=10)
    assert len(air.drone_types) == 10
    for d in air.drone_types:
        assert d.safety_radius >= 50.0
    for r in air.routes:
        assert 2 <= r.num_waypoints <= 4
        for op in (r.takeoff_point, r.landing_point):
            assert 3 <= len(op.pads) <= 5
            for p in op.pads:
                assert 120.0 - 1e-6 <= p.max_dwell <= 240.0 + 1e-6


def test_node_times_monotonic():
    """Thời điểm tới các node tăng dần và các cửa sổ pha nhất quán."""
    rng = np.random.default_rng(1)
    air = build_airspace(rng)
    flights = generate_flight_plans(air, rng, SimParams(num_flights=50))
    for f in flights:
        assert np.all(np.diff(f.node_times) > 0)
        assert f.takeoff_window[1] == f.cruise_window[0]
        assert f.cruise_window[1] == f.landing_window[0]
        assert f.arrival_time == f.landing_window[1]
        # thời gian dừng bãi đỗ không quá pad_t
        assert f.origin_dwell <= f.takeoff_pad.max_dwell + 1e-6
        assert f.dest_dwell <= f.landing_pad.max_dwell + 1e-6


def test_takeoff_conflict_detection():
    """Hai chuyến cùng điểm cất cánh, cửa sổ chồng lấn -> phải phát hiện."""
    rng = np.random.default_rng(2)
    air = build_airspace(rng)
    flights = generate_flight_plans(air, rng, SimParams(num_flights=2))
    # ép hai chuyến cùng đường bay, cùng thời gian cất cánh
    f0, f1 = flights
    f1.route = f0.route
    f1.seg_speeds = f0.seg_speeds.copy()   # khớp số chặng của đường bay mới
    f0.departure_time = 100.0
    f1.departure_time = 105.0
    f0.takeoff_time = f1.takeoff_time = 30.0
    f0.__post_init__()
    f1.__post_init__()
    conf, pairs = C.detect_takeoff_conflicts([f0, f1])
    assert conf == {0, 1}
    assert len(pairs) == 1


def test_enroute_conflict_symmetric_and_none_when_separated():
    """Hai chuyến khác đường bay (tách biệt) không xung đột hành trình."""
    rng = np.random.default_rng(3)
    air = build_airspace(rng)
    # chọn 2 chuyến trên 2 đường bay khác nhau, cùng thời điểm
    params = SimParams(num_flights=40)
    flights = generate_flight_plans(air, rng, params)
    conf, pairs = C.detect_enroute_conflicts(flights, air, dt=1.0)
    # mọi cặp xung đột hành trình phải cùng một đường bay (do các đường tách biệt)
    id2route = {f.id: f.route.id for f in flights}
    for i, j in pairs:
        assert id2route[i] == id2route[j], \
            "Xung đột hành trình chỉ được xảy ra giữa hai drone cùng đường bay"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  OK  {fn.__name__}")
    print(f"\n{len(fns)} kiểm thử đều đạt.")


if __name__ == "__main__":
    _run_all()

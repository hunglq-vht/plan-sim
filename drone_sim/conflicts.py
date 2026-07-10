"""Phát hiện xung đột theo từng pha (chưa có thuật toán xử lý xung đột).

Bốn pha xung đột:

Cất cánh (takeoff)
    Điểm cất cánh hoạt động như một "đường băng" dùng chung: trong lúc một
    phương tiện đang cất cánh, phương tiện khác không được cất/hạ cánh chen vào
    cùng điểm đó. => Xung đột khi hai cửa sổ khai thác tại CÙNG một điểm cất cánh
    chồng lấn thời gian.

Hạ cánh (landing)
    Tương tự, tại cùng một điểm hạ cánh.

Trên hành trình (enroute)
    Hai drone tiến vào khoảng cách nhỏ hơn khoảng cách an toàn yêu cầu
    (max của hai bán kính an toàn, >= 50 m) trong khi cả hai đang ở pha hành
    trình và có thời gian chồng lấn.

Bãi đỗ (parking)
    Hai chuyến bay chiếm CÙNG một bãi đỗ với khoảng thời gian chiếm dụng chồng
    lấn nhau.

Mỗi hàm trả về:
    conflicted : set[int]           -- id các chuyến bay dính xung đột
    pairs      : list[(i, j)]       -- các cặp chuyến bay xung đột
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .airspace import Airspace
from .flight import FlightPlan


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Hai khoảng thời gian nửa mở có giao nhau (dương thực sự) không?"""
    return a[0] < b[1] and b[0] < a[1]


def _sweep_overlaps(
    intervals: Sequence[Tuple[float, float, int]]
) -> List[Tuple[int, int]]:
    """Tìm mọi cặp khoảng chồng lấn bằng thuật toán quét (sweep line).

    ``intervals`` là list các (start, end, flight_id). Trả về list cặp id.
    """
    pairs: List[Tuple[int, int]] = []
    order = sorted(intervals, key=lambda it: it[0])
    active: List[Tuple[float, int]] = []  # (end, flight_id)
    for start, end, fid in order:
        # loại khỏi 'active' các khoảng đã kết thúc trước 'start'
        active = [(e, i) for (e, i) in active if e > start]
        for e, i in active:
            pairs.append((i, fid))
        active.append((end, fid))
    return pairs


def _grouped_window_conflicts(
    flights: Sequence[FlightPlan],
    key_fn,
    window_fn,
) -> Tuple[Set[int], List[Tuple[int, int]]]:
    """Khung chung: nhóm theo ``key_fn`` rồi tìm cửa sổ ``window_fn`` chồng lấn."""
    groups: Dict[int, List[Tuple[float, float, int]]] = {}
    for f in flights:
        w = window_fn(f)
        groups.setdefault(key_fn(f), []).append((w[0], w[1], f.id))

    conflicted: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for items in groups.values():
        if len(items) < 2:
            continue
        for i, j in _sweep_overlaps(items):
            pairs.append((i, j))
            conflicted.add(i)
            conflicted.add(j)
    return conflicted, pairs


# --------------------------------------------------------------------------- #
# Pha Cất cánh / Hạ cánh
# --------------------------------------------------------------------------- #
def detect_takeoff_conflicts(flights):
    return _grouped_window_conflicts(
        flights,
        key_fn=lambda f: f.route.takeoff_point.id,
        window_fn=lambda f: f.takeoff_window,
    )


def detect_landing_conflicts(flights):
    return _grouped_window_conflicts(
        flights,
        key_fn=lambda f: f.route.landing_point.id,
        window_fn=lambda f: f.landing_window,
    )


# --------------------------------------------------------------------------- #
# Pha Bãi đỗ
# --------------------------------------------------------------------------- #
def detect_parking_conflicts(flights):
    """Xung đột khi hai chuyến chiếm cùng một bãi đỗ với thời gian chồng lấn.

    Mỗi chuyến chiếm hai bãi đỗ (cất cánh trước khi đi, hạ cánh sau khi đến).
    Ta gom mọi lần chiếm dụng theo id bãi đỗ.
    """
    occ: Dict[int, List[Tuple[float, float, int]]] = {}
    for f in flights:
        to = f.takeoff_pad_window
        la = f.landing_pad_window
        occ.setdefault(f.takeoff_pad.id, []).append((to[0], to[1], f.id))
        occ.setdefault(f.landing_pad.id, []).append((la[0], la[1], f.id))

    conflicted: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for items in occ.values():
        if len(items) < 2:
            continue
        for i, j in _sweep_overlaps(items):
            if i == j:            # cùng một chuyến chiếm 2 bãi khác nhau -> bỏ qua
                continue
            pairs.append((i, j))
            conflicted.add(i)
            conflicted.add(j)
    return conflicted, pairs


# --------------------------------------------------------------------------- #
# Pha Trên hành trình
# --------------------------------------------------------------------------- #
def detect_enroute_conflicts(
    flights: Sequence[FlightPlan],
    airspace: Airspace,
    dt: float = 1.0,
) -> Tuple[Set[int], List[Tuple[int, int]]]:
    """Xung đột hành trình: khoảng cách < khoảng cách an toàn khi cùng bay.

    Dùng broad-phase (giao thời gian hành trình + giao hộp bao mở rộng) để lọc
    cặp ứng viên, sau đó lấy mẫu vị trí theo bước ``dt`` và so khoảng cách 3D.
    """
    conflicted: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    n = len(flights)

    # Chuẩn bị sẵn cửa sổ hành trình và hộp bao cho từng chuyến
    cruise = [f.cruise_window for f in flights]
    bbox = [f.cruise_bbox() for f in flights]

    for a in range(n):
        fa = flights[a]
        wa = cruise[a]
        amin, amax = bbox[a]
        for b in range(a + 1, n):
            fb = flights[b]
            wb = cruise[b]
            # 1) giao thời gian hành trình
            lo = max(wa[0], wb[0])
            hi = min(wa[1], wb[1])
            if hi - lo <= 0:
                continue
            # 2) giao hộp bao (nới rộng theo khoảng cách an toàn)
            sep = airspace.pairwise_safety(fa.drone_type, fb.drone_type)
            bmin, bmax = bbox[b]
            if np.any(amin - sep > bmax) or np.any(bmin - sep > amax):
                continue
            # 3) lấy mẫu theo thời gian và so khoảng cách
            steps = max(2, int(np.ceil((hi - lo) / dt)) + 1)
            ts = np.linspace(lo, hi, steps)
            pa = fa.positions_at(ts)
            pb = fb.positions_at(ts)
            dmin = float(np.min(np.linalg.norm(pa - pb, axis=1)))
            if dmin < sep:
                pairs.append((fa.id, fb.id))
                conflicted.add(fa.id)
                conflicted.add(fb.id)
    return conflicted, pairs


# --------------------------------------------------------------------------- #
# Tổng hợp một trial
# --------------------------------------------------------------------------- #
def detect_all(
    flights: Sequence[FlightPlan],
    airspace: Airspace,
    dt: float = 1.0,
) -> Dict[str, Set[int]]:
    """Chạy cả bốn pha, trả về dict pha -> tập id chuyến bay dính xung đột."""
    to_c, _ = detect_takeoff_conflicts(flights)
    la_c, _ = detect_landing_conflicts(flights)
    en_c, _ = detect_enroute_conflicts(flights, airspace, dt=dt)
    pk_c, _ = detect_parking_conflicts(flights)
    return {"takeoff": to_c, "landing": la_c, "enroute": en_c, "parking": pk_c}

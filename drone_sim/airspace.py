"""Sinh vùng trời tĩnh cho thí nghiệm.

Vùng trời gồm:
    * ``num_routes`` đường bay KHÔNG giao nhau trong khu vực ``size`` x ``size`` (m).
    * Mỗi đường bay: 1 điểm cất cánh, 1 điểm hạ cánh, 2..4 waypoint.
    * Mỗi điểm cất/hạ cánh có 3..5 bãi đỗ, mỗi bãi đỗ có thời gian dừng tối đa
      ``pad_t`` (2..4 phút).
    * ``num_types`` loại drone, mỗi loại có dải vận tốc và bán kính an toàn
      (>= ``min_safety`` = 50 m).

Bảo đảm KHÔNG giao nhau
-----------------------
Khu vực được chia thành ``num_routes`` dải ngang (strip) không chồng nhau theo
trục y. Mỗi đường bay bị giới hạn hoàn toàn trong dải của nó (toàn bộ waypoint,
điểm cất/hạ cánh, bãi đỗ đều nằm trong phần lõi của dải, cách biên dải một
khoảng ``strip_margin``). Vì các dải cách nhau theo y một khoảng >= 2*margin
(mặc định 200 m) và mọi đường bay đơn điệu theo x, nên hai đường bay bất kỳ
không thể cắt nhau và luôn cách nhau theo phương ngang > 50 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class DroneType:
    """Một loại drone."""

    id: int
    name: str
    speed_min: float          # m/s, vận tốc hành trình tối thiểu
    speed_max: float          # m/s, vận tốc hành trình tối đa
    safety_radius: float      # m, bán kính an toàn tối thiểu của loại này (>= 50)

    def sample_speed(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.speed_min, self.speed_max))


@dataclass
class Pad:
    """Một bãi đỗ tại một điểm cất/hạ cánh."""

    id: int                   # id toàn cục, duy nhất
    point_id: int             # thuộc điểm cất/hạ cánh nào
    kind: str                 # 'takeoff' hoặc 'landing'
    pos: np.ndarray           # (x, y, 0) trên mặt đất
    max_dwell: float          # pad_t: thời gian dừng tối đa cho phép (giây)


@dataclass
class OpPoint:
    """Điểm khai thác: điểm cất cánh hoặc điểm hạ cánh của một đường bay."""

    id: int                   # id toàn cục, duy nhất
    route_id: int
    kind: str                 # 'takeoff' hoặc 'landing'
    pos: np.ndarray           # (x, y, 0)
    pads: List[Pad] = field(default_factory=list)


@dataclass
class Route:
    """Một đường bay: cất cánh -> waypoint... -> hạ cánh."""

    id: int
    altitude: float           # độ cao hành trình (m)
    takeoff_point: OpPoint
    landing_point: OpPoint
    waypoints: np.ndarray     # (k, 3) với k in [2, 4], tất cả ở độ cao 'altitude'

    @property
    def num_waypoints(self) -> int:
        return int(self.waypoints.shape[0])

    def path_nodes(self) -> np.ndarray:
        """Chuỗi node 3D của toàn đường bay: mặt đất -> waypoints -> mặt đất.

        Node[0]  = điểm cất cánh (z = 0)
        Node[1..k] = các waypoint (z = altitude)
        Node[-1] = điểm hạ cánh (z = 0)
        """
        nodes = [self.takeoff_point.pos]
        nodes.extend(list(self.waypoints))
        nodes.append(self.landing_point.pos)
        return np.asarray(nodes, dtype=float)


@dataclass
class Airspace:
    """Toàn bộ vùng trời tĩnh của thí nghiệm."""

    size: float
    routes: List[Route]
    drone_types: List[DroneType]
    min_safety: float = 50.0

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    def pairwise_safety(self, a: DroneType, b: DroneType) -> float:
        """Khoảng cách an toàn yêu cầu giữa hai drone loại a và b."""
        return max(self.min_safety, a.safety_radius, b.safety_radius)


# --------------------------------------------------------------------------- #
# Bộ sinh vùng trời
# --------------------------------------------------------------------------- #
def build_airspace(
    rng: np.random.Generator,
    *,
    num_routes: int = 10,
    num_types: int = 10,
    size: float = 5000.0,
    min_safety: float = 50.0,
    strip_margin: float = 100.0,
    pads_per_point: tuple = (3, 5),
    waypoints_per_route: tuple = (2, 4),
    pad_t_range: tuple = (120.0, 240.0),   # 2..4 phút
    altitude_range: tuple = (60.0, 120.0),
) -> Airspace:
    """Sinh một vùng trời tĩnh (cố định cho mọi trial của Monte Carlo).

    ``rng`` cố định seed để tái lập được cấu hình vùng trời.
    """
    # ---- 10 loại drone: bán kính an toàn trải từ 50 m trở lên ---------------
    drone_types: List[DroneType] = []
    for i in range(num_types):
        # bán kính an toàn 50..95 m; vận tốc hành trình 8..22 m/s tuỳ loại
        safety = float(min_safety + 5.0 * i)          # 50, 55, ..., 50+5*(n-1)
        vmin = float(rng.uniform(8.0, 12.0))
        vmax = float(vmin + rng.uniform(4.0, 10.0))
        drone_types.append(
            DroneType(id=i, name=f"DT{i:02d}", speed_min=vmin,
                      speed_max=vmax, safety_radius=safety)
        )

    strip_h = size / num_routes
    assert strip_h - 2 * strip_margin > 200.0, "Dải quá hẹp so với lề an toàn"

    routes: List[Route] = []
    point_id = 0
    pad_id = 0

    for r in range(num_routes):
        y_lo = r * strip_h + strip_margin
        y_hi = (r + 1) * strip_h - strip_margin
        altitude = float(rng.uniform(*altitude_range))

        # Điểm cất cánh bên trái, điểm hạ cánh bên phải (đường bay đơn điệu theo x)
        x_takeoff = float(rng.uniform(150.0, 400.0))
        x_landing = float(size - rng.uniform(150.0, 400.0))
        y_takeoff = float(rng.uniform(y_lo, y_hi))
        y_landing = float(rng.uniform(y_lo, y_hi))

        takeoff_point = OpPoint(id=point_id, route_id=r, kind="takeoff",
                                pos=np.array([x_takeoff, y_takeoff, 0.0]))
        point_id += 1
        landing_point = OpPoint(id=point_id, route_id=r, kind="landing",
                                pos=np.array([x_landing, y_landing, 0.0]))
        point_id += 1

        # Bãi đỗ cho mỗi điểm
        for op in (takeoff_point, landing_point):
            n_pads = int(rng.integers(pads_per_point[0], pads_per_point[1] + 1))
            for _ in range(n_pads):
                offset = rng.uniform(-30.0, 30.0, size=2)
                pos = np.array([op.pos[0] + offset[0],
                                op.pos[1] + offset[1], 0.0])
                # kẹp trong dải để không rời khu vực đường bay
                pos[1] = float(np.clip(pos[1], y_lo, y_hi))
                max_dwell = float(rng.uniform(*pad_t_range))
                op.pads.append(Pad(id=pad_id, point_id=op.id, kind=op.kind,
                                   pos=pos, max_dwell=max_dwell))
                pad_id += 1

        # Waypoint: x tăng dần giữa cất và hạ cánh, y dao động trong dải
        k = int(rng.integers(waypoints_per_route[0], waypoints_per_route[1] + 1))
        xs = np.linspace(x_takeoff, x_landing, k + 2)[1:-1]  # bỏ hai đầu
        wps = np.zeros((k, 3))
        for j in range(k):
            wps[j, 0] = xs[j]
            wps[j, 1] = float(rng.uniform(y_lo, y_hi))
            wps[j, 2] = altitude

        routes.append(Route(id=r, altitude=altitude,
                            takeoff_point=takeoff_point,
                            landing_point=landing_point, waypoints=wps))

    return Airspace(size=size, routes=routes,
                    drone_types=drone_types, min_safety=min_safety)

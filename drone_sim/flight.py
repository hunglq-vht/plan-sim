"""Kế hoạch bay và quỹ đạo theo thời gian.

Một kế hoạch bay gồm:
    * điểm đi (điểm cất cánh của đường bay) + bãi đỗ cất cánh
    * điểm đến (điểm hạ cánh của đường bay) + bãi đỗ hạ cánh
    * các waypoint (thuộc đường bay)
    * vận tốc từng chặng hành trình
    * thời gian cất cánh (mặt đất -> waypoint đầu)
    * thời gian hạ cánh (waypoint cuối -> bãi đỗ)
    * loại drone
    * thời điểm tạo và thời gian xuất phát (đăng ký sau thời điểm tạo)

Trục thời gian của một chuyến bay
---------------------------------
    t_dep                      : bắt đầu cất cánh tại điểm cất cánh
    t_dep + takeoff_time       : tới waypoint đầu, bắt đầu hành trình
    ... hành trình qua waypoint (mỗi chặng có vận tốc riêng) ...
    t_cruise_end               : tới waypoint cuối, bắt đầu hạ cánh
    t_arr = t_cruise_end + landing_time : chạm bãi đỗ hạ cánh

Chiếm dụng bãi đỗ (pha "Bãi đỗ")
    * bãi đỗ cất cánh: [t_dep - origin_dwell, t_dep]
    * bãi đỗ hạ cánh : [t_arr, t_arr + dest_dwell]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .airspace import DroneType, Pad, Route


@dataclass
class FlightPlan:
    id: int
    route: Route
    drone_type: DroneType

    creation_time: float          # thời điểm tạo kế hoạch bay
    departure_time: float         # thời gian xuất phát đã đăng ký (>= creation_time)

    takeoff_time: float           # giây, mặt đất -> waypoint đầu
    landing_time: float           # giây, waypoint cuối -> bãi đỗ
    seg_speeds: np.ndarray        # (num_cruise_segments,) vận tốc từng chặng (m/s)

    takeoff_pad: Pad
    landing_pad: Pad
    origin_dwell: float           # thời gian chiếm bãi đỗ trước khi cất cánh (<= pad_t)
    dest_dwell: float             # thời gian chiếm bãi đỗ sau khi hạ cánh (<= pad_t)

    # ---- các trường dẫn xuất, tính trong __post_init__ ----
    nodes: np.ndarray = field(init=False)        # (m, 3) node quỹ đạo
    node_times: np.ndarray = field(init=False)   # (m,) thời điểm tới từng node

    def __post_init__(self) -> None:
        self.nodes = self.route.path_nodes()
        self.node_times = self._compute_node_times()

    # ------------------------------------------------------------------ #
    def _compute_node_times(self) -> np.ndarray:
        """Thời điểm (tuyệt đối) drone tới mỗi node của quỹ đạo."""
        nodes = self.nodes
        m = nodes.shape[0]              # = 1 (cất) + k (waypoint) + 1 (hạ)
        times = np.zeros(m)
        times[0] = self.departure_time

        # Chặng cất cánh: node0 -> node1
        times[1] = times[0] + self.takeoff_time

        # Các chặng hành trình: node1 -> node2 -> ... -> node[m-2]
        # số chặng hành trình = (m - 1) - 2 = m - 3  (bằng num_waypoints - 1)
        num_cruise = m - 3
        assert self.seg_speeds.shape[0] == num_cruise, (
            f"seg_speeds phải có {num_cruise} phần tử, có {self.seg_speeds.shape[0]}"
        )
        for i in range(num_cruise):
            a = nodes[1 + i]
            b = nodes[2 + i]
            dist = float(np.linalg.norm(b - a))
            dt = dist / self.seg_speeds[i]
            times[2 + i] = times[1 + i] + dt

        # Chặng hạ cánh: node[m-2] -> node[m-1]
        times[m - 1] = times[m - 2] + self.landing_time
        return times

    # ------------------------------------------------------------------ #
    # Các cửa sổ thời gian theo pha
    # ------------------------------------------------------------------ #
    @property
    def takeoff_window(self) -> Tuple[float, float]:
        return (self.node_times[0], self.node_times[1])

    @property
    def cruise_window(self) -> Tuple[float, float]:
        return (self.node_times[1], self.node_times[-2])

    @property
    def landing_window(self) -> Tuple[float, float]:
        return (self.node_times[-2], self.node_times[-1])

    @property
    def arrival_time(self) -> float:
        return float(self.node_times[-1])

    @property
    def takeoff_pad_window(self) -> Tuple[float, float]:
        return (self.departure_time - self.origin_dwell, self.departure_time)

    @property
    def landing_pad_window(self) -> Tuple[float, float]:
        return (self.arrival_time, self.arrival_time + self.dest_dwell)

    # ------------------------------------------------------------------ #
    # Vị trí theo thời gian
    # ------------------------------------------------------------------ #
    def positions_at(self, ts: np.ndarray) -> np.ndarray:
        """Vị trí 3D tại các thời điểm ``ts`` (nội suy tuyến tính theo quỹ đạo).

        Ngoài khoảng [t_dep, t_arr] vị trí bị kẹp về node đầu/cuối (đứng yên
        trên mặt đất). Vector hoá bằng ``np.interp`` theo từng trục.
        """
        ts = np.asarray(ts, dtype=float)
        x = np.interp(ts, self.node_times, self.nodes[:, 0])
        y = np.interp(ts, self.node_times, self.nodes[:, 1])
        z = np.interp(ts, self.node_times, self.nodes[:, 2])
        return np.stack([x, y, z], axis=-1)

    def cruise_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Hộp bao (min, max) của phần hành trình (các waypoint)."""
        wp = self.route.waypoints
        return wp.min(axis=0), wp.max(axis=0)

"""Mô phỏng Monte Carlo xung đột đường bay drone.

Package cung cấp các thành phần:
    - airspace : sinh vùng trời (đường bay, điểm cất/hạ cánh, bãi đỗ, loại drone)
    - flight   : kế hoạch bay và quỹ đạo theo thời gian
    - conflicts: phát hiện xung đột theo từng pha
    - montecarlo: sinh kế hoạch bay ngẫu nhiên và chạy thí nghiệm Monte Carlo
"""

from .airspace import (
    Airspace,
    Route,
    OpPoint,
    Pad,
    DroneType,
    build_airspace,
)
from .flight import FlightPlan
from .montecarlo import (
    SimParams,
    generate_flight_plans,
    run_trial,
    run_montecarlo,
    PhaseResult,
    MonteCarloResult,
)

__all__ = [
    "Airspace",
    "Route",
    "OpPoint",
    "Pad",
    "DroneType",
    "build_airspace",
    "FlightPlan",
    "SimParams",
    "generate_flight_plans",
    "run_trial",
    "run_montecarlo",
    "PhaseResult",
    "MonteCarloResult",
]

# Tên các pha xung đột (dùng thống nhất trong toàn bộ package)
PHASES = ("takeoff", "landing", "enroute", "parking")
PHASE_LABELS_VI = {
    "takeoff": "Cất cánh",
    "landing": "Hạ cánh",
    "enroute": "Trên hành trình",
    "parking": "Bãi đỗ",
    "any": "Toàn kế hoạch bay",
}

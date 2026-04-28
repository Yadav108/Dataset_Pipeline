from __future__ import annotations

from typing import Any

import numpy as np


class FrameCollector:
    """Collect per-angle RGB+Depth frames and associated ArUco results."""

    def __init__(self, step_angle_deg: float, total_angles: int):
        self.step_angle_deg = float(step_angle_deg)
        self.total_angles = int(total_angles)
        self._frames: dict[int, dict[str, Any]] = {}

    def add_frame(
        self,
        angle_index: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        aruco_result: dict[str, Any],
        resolved_angle_deg: float,
        angle_source: str,
    ) -> None:
        self._frames[int(angle_index)] = {
            "angle_deg": float(angle_index) * self.step_angle_deg,
            "resolved_angle_deg": float(resolved_angle_deg),
            "angle_source": str(angle_source),
            "rgb": rgb,
            "depth": depth,
            "aruco_detected": bool(aruco_result.get("detected", False)),
            "rvec": aruco_result.get("rvec"),
            "tvec": aruco_result.get("tvec"),
        }

    def get_all(self) -> list[dict[str, Any]]:
        return [self._frames[idx] for idx in sorted(self._frames.keys())]

    def is_complete(self) -> bool:
        return len(self._frames) == self.total_angles

    def clear(self) -> None:
        self._frames.clear()


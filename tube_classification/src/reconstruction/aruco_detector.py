from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ArucoDetector:
    """Detect ArUco marker and estimate 6DOF pose in camera coordinates."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, marker_size_m: float):
        self.marker_size_m = float(marker_size_m)
        self.camera_matrix = np.array(
            [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect(self, rgb_frame: np.ndarray) -> dict[str, Any]:
        """Return ArUco detection + pose for one RGB frame."""
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(rgb_frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                rgb_frame, self.dictionary, parameters=self.detector_params
            )

        if ids is None or len(ids) == 0:
            return {
                "detected": False,
                "marker_id": None,
                "corners": None,
                "rvec": None,
                "tvec": None,
            }

        marker_id = int(ids[0][0])
        marker_corners = corners[0]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size_m, self.camera_matrix, self.dist_coeffs
        )

        return {
            "detected": True,
            "marker_id": marker_id,
            "corners": marker_corners,
            "rvec": np.asarray(rvecs[0], dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvecs[0], dtype=np.float64).reshape(3, 1),
        }


from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ReconstructionCaptureLoop:
    """Dedicated timed RGB+Depth capture loop for 3D reconstruction."""

    def __init__(self, streamer, aruco_detector, frame_collector, config):
        self.streamer = streamer
        self.aruco_detector = aruco_detector
        self.frame_collector = frame_collector

        self.rpm = float(config["rpm"])
        self.total_captures = int(config["total_captures"])
        self.step_angle_deg = float(config["step_angle_deg"])
        self.capture_interval_sec = float(config["capture_interval_sec"])

    def run(self):
        # Step 1 — Origin calibration: wait until ArUco is visible.
        origin_rvec = None
        origin_tvec = None
        origin_time = None

        while True:
            frames = self.streamer.get_aligned_frames()
            if frames is None:
                continue

            rgb, _ = frames
            aruco_result = self.aruco_detector.detect(rgb)
            if aruco_result.get("detected", False):
                origin_rvec = np.asarray(aruco_result["rvec"], dtype=np.float64).reshape(3, 1)
                origin_tvec = np.asarray(aruco_result["tvec"], dtype=np.float64).reshape(3, 1)
                origin_time = time.time()
                logger.info("Origin frame captured. Starting reconstruction loop.")
                break

        # Keep variables explicit for traceability/logging, even if origin_tvec is not directly used.
        _ = origin_tvec
        _ = origin_time
        time.sleep(0.5)

        # Step 2 — Timed capture loop.
        aruco_resolved = 0
        fallback_resolved = 0

        for capture_index in range(self.total_captures):
            capture_start = time.time()

            frames = self.streamer.get_aligned_frames()
            while frames is None:
                frames = self.streamer.get_aligned_frames()
            rgb, depth = frames

            aruco_result = self.aruco_detector.detect(rgb)

            if aruco_result.get("detected", False):
                current_rvec = np.asarray(aruco_result["rvec"], dtype=np.float64).reshape(3, 1)
                current_R, _ = cv2.Rodrigues(current_rvec)
                origin_R, _ = cv2.Rodrigues(origin_rvec)
                relative_R = current_R @ origin_R.T
                angle_deg = math.degrees(math.atan2(float(relative_R[2, 0]), float(relative_R[0, 0])))
                angle_source = "aruco"
                aruco_resolved += 1
            else:
                angle_deg = float(capture_index) * self.step_angle_deg
                angle_source = "fallback"
                fallback_resolved += 1

            self.frame_collector.add_frame(
                angle_index=capture_index,
                rgb=rgb,
                depth=depth,
                aruco_result=aruco_result,
                resolved_angle_deg=angle_deg,
                angle_source=angle_source,
            )

            logger.info(
                "Capture %d/%d | Angle: %.1f° (%s) | ArUco: %s",
                capture_index + 1,
                self.total_captures,
                angle_deg,
                angle_source,
                aruco_result.get("detected", False),
            )

            elapsed = time.time() - capture_start
            sleep_time = max(0.0, self.capture_interval_sec - elapsed)
            time.sleep(sleep_time)

        logger.info(
            "Capture loop complete. ArUco resolved: %d/%d, Fallback: %d/%d",
            aruco_resolved,
            self.total_captures,
            fallback_resolved,
            self.total_captures,
        )
        return self.frame_collector


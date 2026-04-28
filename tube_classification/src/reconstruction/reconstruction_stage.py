from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.reconstruction.aruco_detector import ArucoDetector
from src.reconstruction.cloud_registrar import CloudRegistrar
from src.reconstruction.point_cloud_builder import PointCloudBuilder


logger = logging.getLogger(__name__)


class ReconstructionStage:
    """Run end-to-end reconstruction from collected RGB+Depth frames."""

    def __init__(self, config: dict):
        self.step_angle_deg = float(config["step_angle_deg"])
        self.total_angles = int(config["total_angles"])
        self.marker_size_m = float(config["marker_size_m"])
        self.output_dir = Path(config["output_dir"])

    def run(self, frames: list, intrinsics: dict) -> str:
        import open3d as o3d

        if len(frames) == 0:
            logger.warning("Reconstruction skipped: no frames provided")
            return ""

        detector = ArucoDetector(
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            marker_size_m=self.marker_size_m,
        )
        builder = PointCloudBuilder(
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            depth_scale=float(intrinsics["depth_scale"]),
        )
        registrar = CloudRegistrar(step_angle_deg=self.step_angle_deg)

        registered_frames: list[dict[str, Any]] = []
        aruco_detected_count = 0

        for frame in frames:
            rgb = frame["rgb"]
            depth = frame["depth"]
            angle_deg = float(frame.get("angle_deg", 0.0))

            aruco_result = detector.detect(rgb)
            if aruco_result["detected"]:
                aruco_detected_count += 1

            cloud = builder.build(rgb, depth)
            registered_frames.append(
                {
                    "angle_deg": angle_deg,
                    "cloud": cloud,
                    "aruco_detected": aruco_result["detected"],
                    "tvec": aruco_result["tvec"],
                }
            )

        merged_cloud = registrar.register(registered_frames)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "reconstruction.ply"
        o3d.io.write_point_cloud(str(output_path), merged_cloud)

        logger.info(
            "Reconstruction complete: ArUco detected in %d/%d frames",
            aruco_detected_count,
            len(frames),
        )

        return str(output_path)


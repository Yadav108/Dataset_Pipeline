from __future__ import annotations

import numpy as np


class CloudRegistrar:
    """Register per-angle point clouds into a single merged cloud."""

    def __init__(self, step_angle_deg: float):
        self.step_angle_deg = float(step_angle_deg)

    def register(self, frames: list) -> o3d.geometry.PointCloud:
        import open3d as o3d

        merged = o3d.geometry.PointCloud()

        for frame in frames:
            cloud = frame.get("cloud")
            if cloud is None:
                continue

            transformed = o3d.geometry.PointCloud(cloud)
            angle_deg = float(frame.get("resolved_angle_deg", 0.0))
            angle_rad = np.deg2rad(angle_deg)
            rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.0, angle_rad, 0.0))
            transformed.rotate(rotation, center=(0.0, 0.0, 0.0))

            if bool(frame.get("aruco_detected", False)) and frame.get("tvec") is not None:
                tvec = np.asarray(frame["tvec"], dtype=np.float64).reshape(-1)
                if tvec.size >= 3:
                    transformed.translate((float(tvec[0]), float(tvec[1]), float(tvec[2])), relative=True)

            merged += transformed

        return merged


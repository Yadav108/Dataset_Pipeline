from __future__ import annotations

import cv2
import numpy as np


class PointCloudBuilder:
    """Build colored point clouds from aligned RGB+Depth frames."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
        depth_trunc: float = 1.0,
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.depth_scale = float(depth_scale)
        self.depth_trunc = float(depth_trunc)

        # Open3D expects: depth_in_meters = depth_value / depth_scale
        # RealSense depth_scale is typically meters per unit (e.g. 0.001), so invert it.
        self.o3d_depth_scale = (
            (1.0 / self.depth_scale) if self.depth_scale > 0.0 and self.depth_scale < 1.0 else self.depth_scale
        )

    def build(self, rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
        import open3d as o3d

        color_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        color_img = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16)))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_img,
            depth=depth_img,
            depth_scale=self.o3d_depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        height, width = depth.shape[:2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            int(width), int(height), self.fx, self.fy, self.cx, self.cy
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)


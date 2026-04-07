"""Live preview renderer for overhead rack view with grid overlay and occupied slot detection."""

import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Set, Tuple, Optional, Any


@dataclass
class PreviewResult:
    """Result from render() call."""
    frame: np.ndarray
    occupied_slots: Set[Tuple[int, int]]
    user_input: Optional[str]
    timestamp: datetime
    fps: float


class LivePreviewRenderer:
    """Renders overhead rack view with grid overlay and operator controls."""

    def __init__(self, grid_map: Dict[str, Any], calib_params: Dict[str, Any], resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize preview renderer.

        Args:
            grid_map: Grid configuration with slot positions, rows, cols
            calib_params: Calibration parameters including ArUco markers, depth baseline
            resolution: Frame resolution (width, height)
        """
        self.grid_map = grid_map
        self.calib_params = calib_params
        self.width, self.height = resolution
        self.rows = grid_map.get('rows', 5)
        self.cols = grid_map.get('cols', 10)
        self.depth_baseline_mm = calib_params.get('depth_baseline_mm', 330)
        
        # FPS tracking
        self.frame_times = []

    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame timings."""
        if len(self.frame_times) < 2:
            return 0.0
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
        return len(self.frame_times) / time_diff

    def _detect_occupied_slots(self, depth_frame: np.ndarray, depth_tolerance_mm: float = 20.0) -> Set[Tuple[int, int]]:
        """Detect occupied slots by analyzing depth frame."""
        occupied = set()
        
        # Convert depth to millimeters if needed
        depth_mm = depth_frame.astype(np.float32)
        if np.max(depth_mm) > 10000:
            depth_mm = depth_mm / 1000.0
        
        # Threshold for occupation
        depth_threshold = self.depth_baseline_mm - depth_tolerance_mm
        occupied_pixels = depth_mm < depth_threshold
        occupied_pixels = occupied_pixels & (depth_mm > 0)
        
        # Group occupied pixels by grid slot
        slot_height = self.height / self.rows
        slot_width = self.width / self.cols
        
        for row in range(self.rows):
            for col in range(self.cols):
                y_start = int(row * slot_height)
                y_end = int((row + 1) * slot_height)
                x_start = int(col * slot_width)
                x_end = int((col + 1) * slot_width)
                
                slot_region = occupied_pixels[y_start:y_end, x_start:x_end]
                if np.sum(slot_region) > 50:
                    occupied.add((row, col))
        
        return occupied

    def _draw_grid_overlay(self, frame: np.ndarray, occupied_slots: Set[Tuple[int, int]], 
                          grid_line_alpha: float = 0.3,
                          occupied_color: Tuple[int, int, int] = (0, 255, 0),
                          empty_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        """Draw grid lines and slot labels on frame."""
        frame = frame.copy()
        slot_height = self.height / self.rows
        slot_width = self.width / self.cols
        
        overlay = frame.copy()
        
        # Vertical lines
        for col in range(self.cols + 1):
            x = int(col * slot_width)
            cv2.line(overlay, (x, 0), (x, self.height), (255, 255, 255), 1)
        
        # Horizontal lines
        for row in range(self.rows + 1):
            y = int(row * slot_height)
            cv2.line(overlay, (0, y), (self.width, y), (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, grid_line_alpha, frame, 1 - grid_line_alpha, 0, frame)
        
        # Draw slot rectangles
        for row in range(self.rows):
            for col in range(self.cols):
                x_start = int(col * slot_width)
                y_start = int(row * slot_height)
                x_end = int((col + 1) * slot_width)
                y_end = int((row + 1) * slot_height)
                
                color = occupied_color if (row, col) in occupied_slots else empty_color
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                
                label = f"{row},{col}"
                cv2.putText(frame, label, (x_start + 5, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def _draw_aruco_markers(self, frame: np.ndarray, marker_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw ArUco marker positions as circles."""
        frame = frame.copy()
        markers = self.calib_params.get('aruco_markers', [])
        
        for marker in markers:
            if 'position_px' in marker:
                x, y = marker['position_px']
                cv2.circle(frame, (int(x), int(y)), 10, marker_color, 2)
                cv2.circle(frame, (int(x), int(y)), 3, marker_color, -1)
        
        return frame

    def _draw_status_bar(self, frame: np.ndarray, occupied_count: int, fps: float,
                        calib_status: str = "LOADED") -> np.ndarray:
        """Draw status bar at bottom of frame."""
        frame = frame.copy()
        
        status_height = 30
        cv2.rectangle(frame, (0, self.height - status_height), (self.width, self.height), (0, 0, 0), -1)
        
        status_text = (
            f"[LIVE] Mode: -- | Calib: {calib_status} | "
            f"Depth baseline: {self.depth_baseline_mm}mm | "
            f"Occupied: {occupied_count}/{self.rows * self.cols} | "
            f"FPS: {fps:.1f}"
        )
        
        cv2.putText(frame, status_text, (10, self.height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def render(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, 
               config: Optional[Dict[str, Any]] = None) -> PreviewResult:
        """
        Render live preview with overlays and check for keyboard input.

        Args:
            rgb_frame: RGB frame from camera (BGR format)
            depth_frame: Depth frame from camera (mm or raw units)
            config: Configuration dict with preview parameters

        Returns:
            PreviewResult containing annotated frame, occupied slots, user input, etc.
        """
        config = config or {}
        depth_tolerance_mm = config.get('depth_tolerance_mm', 20.0)
        grid_line_alpha = config.get('grid_line_alpha', 0.3)
        occupied_color = tuple(config.get('occupied_color', [0, 255, 0]))
        empty_color = tuple(config.get('empty_color', [128, 128, 128]))
        
        if rgb_frame.shape[:2] != (self.height, self.width):
            rgb_frame = cv2.resize(rgb_frame, (self.width, self.height))
        if depth_frame.shape[:2] != (self.height, self.width):
            depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        occupied_slots = self._detect_occupied_slots(depth_frame, depth_tolerance_mm)
        
        annotated_frame = self._draw_grid_overlay(rgb_frame, occupied_slots, grid_line_alpha, occupied_color, empty_color)
        annotated_frame = self._draw_aruco_markers(annotated_frame)
        
        current_time = datetime.now().timestamp()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        fps = self._calculate_fps()
        
        annotated_frame = self._draw_status_bar(annotated_frame, len(occupied_slots), fps)
        
        cv2.imshow("Tube Classification - Live Preview", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        user_input = None
        if key == ord('s') or key == ord('S'):
            user_input = 'S'
        elif key == ord('b') or key == ord('B'):
            user_input = 'B'
        elif key == ord('c') or key == ord('C'):
            user_input = 'C'
        elif key == ord('q') or key == ord('Q'):
            user_input = 'Q'
        
        return PreviewResult(
            frame=annotated_frame,
            occupied_slots=occupied_slots,
            user_input=user_input,
            timestamp=datetime.now(),
            fps=fps,
        )

    def cleanup(self):
        """Cleanup resources."""
        cv2.destroyAllWindows()

"""Per-slot confirmation preview with quality metrics."""

import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from enum import Enum


class ConfirmationAction(str, Enum):
    """Operator actions in confirmation preview."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    RETAKE = "RETAKE"
    QUIT = "QUIT"


@dataclass
class QualityMetrics:
    """Quality metrics for captured frame."""
    depth_mean: float
    depth_variance: float
    depth_stable: bool
    blur_score: float
    blur_pass: bool
    mask_confidence: float
    mask_pass: bool
    overall_quality: str  # GOOD, FAIR, POOR


@dataclass
class ConfirmationResult:
    """Result from confirmation preview."""
    action: ConfirmationAction
    metrics: QualityMetrics
    timestamp: datetime


class ConfirmationPreviewRenderer:
    """Renders per-slot confirmation preview with quality metrics."""

    def __init__(self, grid_map: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize confirmation renderer.

        Args:
            grid_map: Grid configuration with rows, cols
            config: Configuration dict with confirmation parameters
        """
        self.grid_map = grid_map
        self.config = config or {}
        self.rows = grid_map.get('rows', 5)
        self.cols = grid_map.get('cols', 10)
        
        # Config parameters
        self.zoom_factor = self.config.get('zoom_factor', 4)
        self.depth_stable_variance_mm = self.config.get('depth_stable_variance_mm', 5.0)
        self.depth_expected_mm = self.config.get('depth_expected_mm', 330)
        self.blur_threshold = self.config.get('blur_threshold', 100.0)
        self.mask_confidence_threshold = self.config.get('mask_confidence_threshold', 0.85)
        self.panel_width = self.config.get('panel_width_px', 300)

    def _get_slot_roi(self, frame: np.ndarray, slot_id: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get ROI (x1, y1, x2, y2) for a slot."""
        row, col = slot_id
        height, width = frame.shape[:2]
        
        slot_height = height / self.rows
        slot_width = width / self.cols
        
        x1 = int(col * slot_width)
        y1 = int(row * slot_height)
        x2 = int((col + 1) * slot_width)
        y2 = int((row + 1) * slot_height)
        
        return x1, y1, x2, y2

    def _extract_roi(self, frame: np.ndarray, slot_id: Tuple[int, int]) -> np.ndarray:
        """Extract ROI for a slot."""
        x1, y1, x2, y2 = self._get_slot_roi(frame, slot_id)
        return frame[y1:y2, x1:x2]

    def _calculate_blur_score(self, rgb_roi: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        if rgb_roi.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _calculate_depth_metrics(self, depth_roi: np.ndarray) -> Tuple[float, float, bool]:
        """Calculate depth metrics."""
        depth_roi = depth_roi.astype(np.float32)
        
        # Convert to mm if needed
        if np.max(depth_roi) > 10000:
            depth_roi = depth_roi / 1000.0
        
        # Exclude invalid depths
        valid = depth_roi[depth_roi > 0]
        if len(valid) == 0:
            return 0.0, float('inf'), False
        
        mean = float(np.mean(valid))
        variance = float(np.var(valid))
        stable = variance < self.depth_stable_variance_mm
        
        return mean, variance, stable

    def _calculate_mask_confidence(self, mask: np.ndarray) -> float:
        """Calculate mask confidence (% of foreground)."""
        if mask.size == 0:
            return 0.0
        
        foreground = np.sum(mask > 127)
        total = mask.size
        return float(foreground / total)

    def _calculate_quality_metrics(self, rgb_roi: np.ndarray, depth_roi: np.ndarray, 
                                   mask_roi: np.ndarray) -> QualityMetrics:
        """Calculate all quality metrics."""
        # Depth metrics
        depth_mean, depth_variance, depth_stable = self._calculate_depth_metrics(depth_roi)
        
        # Blur score
        blur_score = self._calculate_blur_score(rgb_roi)
        blur_pass = blur_score > self.blur_threshold
        
        # Mask confidence
        mask_confidence = self._calculate_mask_confidence(mask_roi)
        mask_pass = mask_confidence > self.mask_confidence_threshold
        
        # Overall quality
        if blur_pass and depth_stable and mask_pass:
            overall_quality = "GOOD"
        elif blur_score > self.blur_threshold * 0.5 and depth_variance < self.depth_stable_variance_mm * 2:
            overall_quality = "FAIR"
        else:
            overall_quality = "POOR"
        
        return QualityMetrics(
            depth_mean=depth_mean,
            depth_variance=depth_variance,
            depth_stable=depth_stable,
            blur_score=blur_score,
            blur_pass=blur_pass,
            mask_confidence=mask_confidence,
            mask_pass=mask_pass,
            overall_quality=overall_quality,
        )

    def _draw_zoomed_roi(self, rgb_roi: np.ndarray, mask_roi: np.ndarray) -> np.ndarray:
        """Draw zoomed ROI with mask overlay."""
        # Resize with zoom factor
        zoomed_rgb = cv2.resize(rgb_roi, None, fx=self.zoom_factor, fy=self.zoom_factor, 
                               interpolation=cv2.INTER_LINEAR)
        zoomed_mask = cv2.resize(mask_roi, None, fx=self.zoom_factor, fy=self.zoom_factor, 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        overlay = zoomed_rgb.copy()
        mask_overlay = cv2.cvtColor(zoomed_mask, cv2.COLOR_GRAY2BGR)
        mask_overlay[:, :, 0] = 0  # No blue
        mask_overlay[:, :, 2] = 0  # No red
        
        # Blend mask (green translucent)
        result = cv2.addWeighted(zoomed_rgb, 0.7, mask_overlay * 0.3, 1.0, 0)
        
        return result

    def _draw_metrics_panel(self, metrics: QualityMetrics, panel_height: int = 400) -> np.ndarray:
        """Draw metrics panel."""
        panel = np.zeros((panel_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark background
        
        y_offset = 20
        line_height = 30
        
        # Helper to draw metric with color coding
        def draw_metric(label: str, value: str, passes: bool, y: int):
            color = (0, 255, 0) if passes else (0, 165, 255)  # Green or Orange
            cv2.putText(panel, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(panel, value, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Depth metrics
        depth_text = f"{metrics.depth_mean:.1f}mm (±{metrics.depth_variance:.1f})"
        draw_metric("Depth:", depth_text, metrics.depth_stable, y_offset)
        y_offset += line_height
        
        # Blur score
        blur_text = f"{metrics.blur_score:.1f}"
        draw_metric("Blur:", blur_text, metrics.blur_pass, y_offset)
        y_offset += line_height
        
        # Mask confidence
        mask_text = f"{metrics.mask_confidence * 100:.1f}%"
        draw_metric("Mask:", mask_text, metrics.mask_pass, y_offset)
        y_offset += line_height
        
        # Overall quality
        quality_colors = {
            'GOOD': (0, 255, 0),
            'FAIR': (0, 165, 255),
            'POOR': (0, 0, 255),
        }
        color = quality_colors.get(metrics.overall_quality, (255, 255, 255))
        cv2.putText(panel, "Quality:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, metrics.overall_quality, (10, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return panel

    def _draw_action_bar(self, width: int, height: int = 50) -> np.ndarray:
        """Draw action bar at bottom."""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        bar[:] = (0, 0, 0)
        
        text = "[A] ACCEPT  [R] REJECT  [T] RETAKE  [Q] QUIT"
        cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bar

    def show_confirmation(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, 
                         mask_frame: np.ndarray, slot_id: Tuple[int, int], 
                         capture_index: int, total_captures: int) -> ConfirmationResult:
        """
        Show confirmation preview for a slot.

        Args:
            rgb_frame: RGB frame
            depth_frame: Depth frame
            mask_frame: Segmentation mask from MobileSAM
            slot_id: (row, col) of slot
            capture_index: Current capture number
            total_captures: Total captures for this batch

        Returns:
            ConfirmationResult with operator's action and metrics
        """
        # Extract ROIs
        rgb_roi = self._extract_roi(rgb_frame, slot_id)
        depth_roi = self._extract_roi(depth_frame, slot_id)
        mask_roi = self._extract_roi(mask_frame, slot_id)
        
        # Calculate metrics
        metrics = self._calculate_quality_metrics(rgb_roi, depth_roi, mask_roi)
        
        # Draw components
        zoomed_view = self._draw_zoomed_roi(rgb_roi, mask_roi)
        metrics_panel = self._draw_metrics_panel(metrics, zoomed_view.shape[0])
        action_bar = self._draw_action_bar(zoomed_view.shape[1] + self.panel_width)
        
        # Combine zoomed view and metrics side-by-side
        # Pad zoomed view if needed
        if zoomed_view.shape[0] < metrics_panel.shape[0]:
            pad = metrics_panel.shape[0] - zoomed_view.shape[0]
            zoomed_view = cv2.vconcat([zoomed_view, np.zeros((pad, zoomed_view.shape[1], 3), dtype=np.uint8)])
        
        top_row = cv2.hconcat([zoomed_view, metrics_panel])
        display = cv2.vconcat([top_row, action_bar])
        
        # Add title
        title = f"Slot {slot_id} | Capture {capture_index}/{total_captures}"
        cv2.putText(display, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Confirmation Preview", display)
        
        # Wait for user input
        action = ConfirmationAction.ACCEPT
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a') or key == ord('A'):
                action = ConfirmationAction.ACCEPT
                break
            elif key == ord('r') or key == ord('R'):
                action = ConfirmationAction.REJECT
                break
            elif key == ord('t') or key == ord('T'):
                action = ConfirmationAction.RETAKE
                break
            elif key == ord('q') or key == ord('Q'):
                action = ConfirmationAction.QUIT
                break
        
        cv2.destroyWindow("Confirmation Preview")
        
        return ConfirmationResult(
            action=action,
            metrics=metrics,
            timestamp=datetime.now(),
        )

    def cleanup(self):
        """Cleanup resources."""
        cv2.destroyAllWindows()

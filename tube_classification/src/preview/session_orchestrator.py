"""Session orchestrator - master workflow for capture pipeline."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import numpy as np
import cv2

from .live_preview import LivePreviewRenderer
from .capture_workflow import CaptureWorkflow, CaptureMode
from .confirmation_preview import ConfirmationPreviewRenderer, ConfirmationAction


@dataclass
class CaptureRecord:
    """Record of a single capture in session."""
    slot_id: Tuple[int, int]
    capture_index: int
    action: str
    metrics: Dict[str, Any]
    timestamp: str
    rgb_path: str
    depth_path: str
    mask_path: str
    metadata_path: str


@dataclass
class SessionState:
    """Complete session state."""
    session_id: str
    start_time: str
    mode: str
    slots_selected: List[Tuple[int, int]]
    captures: List[CaptureRecord]
    end_time: Optional[str] = None
    summary: Optional[Dict[str, int]] = None


class CaptureSessionOrchestrator:
    """Master orchestrator for capture session."""

    def __init__(self, camera_manager: Any, grid_map: Dict[str, Any], 
                 calib_params: Dict[str, Any], config: Dict[str, Any],
                 session_id: Optional[str] = None):
        """
        Initialize session orchestrator.

        Args:
            camera_manager: Camera manager instance
            grid_map: Grid configuration
            calib_params: Calibration parameters
            config: Full configuration
            session_id: Override session ID (for resume)
        """
        self.camera = camera_manager
        self.grid_map = grid_map
        self.calib_params = calib_params
        self.config = config
        
        # Session setup
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(config.get('session_dir', 'dataset/sessions')) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.preview = LivePreviewRenderer(grid_map, calib_params)
        self.workflow = CaptureWorkflow(grid_map, config.get('capture', {}))
        self.confirmation = ConfirmationPreviewRenderer(grid_map, config.get('confirmation', {}))
        
        # Session state
        self.state = SessionState(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            mode='',
            slots_selected=[],
            captures=[],
        )
        
        self.logger.info(f"Session initialized: {self.session_id}")

    def _setup_logging(self) -> logging.Logger:
        """Setup session logging."""
        logger = logging.getLogger(f"session_{self.session_id}")
        logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(self.session_dir / 'session.log')
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _capture_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture frame from camera."""
        rgb, depth = self.camera.get_frame()
        return rgb, depth

    def _run_live_preview(self) -> Optional[str]:
        """Run live preview loop until user selects mode."""
        self.logger.info("Starting live preview loop")
        print("\n[LIVE PREVIEW]")
        print("Press [S] for SINGLE, [B] for BATCH, [C] for calibration, [Q] to quit")
        
        # Check if in dry-run mode (no camera)
        if self.camera is None:
            print("\n[DRY-RUN] Simulating mode selection...")
            print("Simulating: User presses [B] for BATCH")
            self.logger.info("Dry-run: Simulating BATCH mode selection")
            return 'B'
        
        try:
            while True:
                rgb, depth = self._capture_frame()
                preview_config = self.config.get('preview', {})
                result = self.preview.render(rgb, depth, preview_config)
                
                if result.user_input:
                    self.logger.info(f"User input: {result.user_input}")
                    return result.user_input
                    
        except KeyboardInterrupt:
            self.logger.info("Live preview interrupted")
            return None
        finally:
            self.preview.cleanup()

    def _run_capture_loop(self, slots_to_capture: List[Tuple[int, int]]) -> None:
        """Run confirmation loop for each slot."""
        self.logger.info(f"Starting capture loop for {len(slots_to_capture)} slots")
        
        # Check if in dry-run mode
        if self.camera is None:
            print("\n[DRY-RUN] Simulating capture loop...")
            for i, slot_id in enumerate(slots_to_capture, 1):
                print(f"  [SIMULATE] Slot {slot_id} - Simulating capture {i}/{len(slots_to_capture)}")
                # Simulate ACCEPT action
                from dataclasses import dataclass
                @dataclass
                class FakeMetrics:
                    depth_mean: float = 330.0
                    depth_variance: float = 2.0
                    blur_score: float = 145.0
                    mask_confidence: float = 0.92
                    overall_quality: str = "GOOD"
                
                fake_metrics = FakeMetrics()
                # Just count as accepted in dry-run
                self.state.captures.append(type('obj', (object,), {
                    'slot_id': slot_id,
                    'capture_index': i,
                    'action': 'ACCEPT',
                    'metrics': {},
                    'timestamp': datetime.now().isoformat(),
                    'rgb_path': f'captures/capture_{i:04d}.jpg',
                    'depth_path': f'captures/capture_{i:04d}_depth.npz',
                    'mask_path': f'captures/capture_{i:04d}_mask.png',
                    'metadata_path': f'captures/capture_{i:04d}.json',
                })())
            print(f"  ✓ Simulated {len(slots_to_capture)} captures")
            return
        
        capture_index = 1
        total = len(slots_to_capture)
        
        for slot_id in slots_to_capture:
            self.logger.info(f"Capturing slot {slot_id} ({capture_index}/{total})")
            print(f"\n[CAPTURE] Slot {slot_id} ({capture_index}/{total})")
            
            # Capture frames
            rgb, depth = self._capture_frame()
            
            # Run segmentation (dummy for now - would call MobileSAM)
            mask = np.zeros_like(depth, dtype=np.uint8)
            
            # Show confirmation
            try:
                confirmation = self.confirmation.show_confirmation(
                    rgb, depth, mask, slot_id, capture_index, total
                )
                
                self.logger.info(f"Operator action: {confirmation.action} | Quality: {confirmation.metrics.overall_quality}")
                
                if confirmation.action == ConfirmationAction.ACCEPT:
                    self._save_capture(slot_id, capture_index, rgb, depth, mask, confirmation.metrics)
                    print(f"  ✓ Slot {slot_id} ACCEPTED")
                    capture_index += 1
                elif confirmation.action == ConfirmationAction.REJECT:
                    self.logger.info(f"Slot {slot_id} REJECTED")
                    print(f"  ✗ Slot {slot_id} REJECTED")
                    capture_index += 1
                elif confirmation.action == ConfirmationAction.RETAKE:
                    self.logger.info(f"Slot {slot_id} RETAKE requested")
                    print(f"  ↻ Slot {slot_id} RETAKE")
                    # Loop will retry this slot
                    slots_to_capture.append(slot_id)  # Re-add to end of list
                elif confirmation.action == ConfirmationAction.QUIT:
                    self.logger.info("User quit session")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error during confirmation: {e}")
                print(f"  ✗ Error: {e}")
                capture_index += 1

    def _save_capture(self, slot_id: Tuple[int, int], capture_index: int,
                     rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray,
                     metrics: Any) -> None:
        """Save capture files."""
        captures_dir = self.session_dir / 'captures'
        captures_dir.mkdir(exist_ok=True)
        
        prefix = f"capture_{capture_index:04d}"
        
        # Save RGB
        rgb_path = captures_dir / f"{prefix}.jpg"
        cv2.imwrite(str(rgb_path), rgb)
        
        # Save depth
        depth_path = captures_dir / f"{prefix}_depth.npz"
        np.savez(depth_path, depth=depth)
        
        # Save mask
        mask_path = captures_dir / f"{prefix}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Save metadata
        metadata = {
            'slot_id': slot_id,
            'capture_index': capture_index,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'depth_mean': float(metrics.depth_mean),
                'depth_variance': float(metrics.depth_variance),
                'blur_score': float(metrics.blur_score),
                'mask_confidence': float(metrics.mask_confidence),
                'overall_quality': metrics.overall_quality,
            }
        }
        metadata_path = captures_dir / f"{prefix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Record capture
        record = CaptureRecord(
            slot_id=slot_id,
            capture_index=capture_index,
            action='ACCEPT',
            metrics=metadata['metrics'],
            timestamp=metadata['timestamp'],
            rgb_path=str(rgb_path.relative_to(self.session_dir)),
            depth_path=str(depth_path.relative_to(self.session_dir)),
            mask_path=str(mask_path.relative_to(self.session_dir)),
            metadata_path=str(metadata_path.relative_to(self.session_dir)),
        )
        self.state.captures.append(record)
        self.logger.info(f"Capture saved: {prefix}")

    def run_session(self) -> Dict[str, Any]:
        """
        Run complete capture session.

        Returns:
            Summary dict with accepted, rejected, session_path
        """
        try:
            # Live preview
            user_input = self._run_live_preview()
            if user_input == 'Q' or user_input is None:
                self.logger.info("Session cancelled by user")
                return self._finalize_session()
            
            # Get capture mode and slots
            preview_config = self.config.get('preview', {})
            
            # Handle dry-run mode for slot selection
            if self.camera is None:
                print("\n[DRY-RUN] Simulating slot selection...")
                # Simulate some occupied slots for dry-run
                occupied_slots = {(1, 2), (2, 3), (3, 4)}
                workflow_result = self.workflow.get_slots_to_capture(occupied_slots)
                if not workflow_result:
                    self.logger.info("Workflow cancelled in dry-run")
                    return self._finalize_session()
            else:
                while True:
                    rgb, depth = self._capture_frame()
                    preview_result = self.preview.render(rgb, depth, preview_config)
                    occupied_slots = preview_result.occupied_slots
                    
                    workflow_result = self.workflow.get_slots_to_capture(occupied_slots)
                    if workflow_result:
                        break
                    else:
                        self.logger.info("Workflow cancelled, returning to preview")
            
            self.state.mode = workflow_result.mode.value
            self.state.slots_selected = workflow_result.slots
            self.logger.info(f"Workflow result: mode={self.state.mode}, slots={len(self.state.slots_selected)}")
            
            # Run capture loop
            self._run_capture_loop(workflow_result.slots.copy())
            
        except KeyboardInterrupt:
            self.logger.info("Session interrupted by user")
        except Exception as e:
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            self.confirmation.cleanup()
        
        return self._finalize_session()

    def _finalize_session(self) -> Dict[str, Any]:
        """Finalize session and save state."""
        self.state.end_time = datetime.now().isoformat()
        
        # Count results
        accepted = sum(1 for c in self.state.captures if c.action == 'ACCEPT')
        total = len(self.state.captures)
        rejected = total - accepted
        
        self.state.summary = {
            'accepted': accepted,
            'rejected': rejected,
            'total': total,
        }
        
        # Save session state
        state_path = self.session_dir / 'metadata.json'
        with open(state_path, 'w') as f:
            json.dump(
                {
                    'session_id': self.state.session_id,
                    'start_time': self.state.start_time,
                    'end_time': self.state.end_time,
                    'mode': self.state.mode,
                    'slots_selected': self.state.slots_selected,
                    'summary': self.state.summary,
                    'captures': [asdict(c) for c in self.state.captures],
                },
                f, indent=2
            )
        
        self.logger.info(f"Session finalized: {self.state.summary}")
        
        print(f"\n[SESSION SUMMARY]")
        print(f"  ✓ Accepted: {accepted}")
        print(f"  ✗ Rejected: {rejected}")
        print(f"  Total: {total}")
        print(f"  Location: {self.session_dir}")
        
        return {
            'session_id': self.state.session_id,
            'accepted': accepted,
            'rejected': rejected,
            'session_path': str(self.session_dir),
        }

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.camera.stop()
        except:
            pass
        self.preview.cleanup()
        self.confirmation.cleanup()

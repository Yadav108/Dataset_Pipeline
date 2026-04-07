"""Capture mode state machine for SINGLE vs BATCH workflow."""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class CaptureMode(str, Enum):
    """Capture mode options."""
    SINGLE = "SINGLE"
    BATCH = "BATCH"


@dataclass
class CaptureWorkflowResult:
    """Result from capture workflow."""
    mode: CaptureMode
    slots: List[Tuple[int, int]]
    count: int
    user_input_history: List[str]


class CaptureWorkflow:
    """State machine for SINGLE vs BATCH mode selection and slot input."""

    def __init__(self, grid_map: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow.

        Args:
            grid_map: Grid configuration with rows, cols
            config: Configuration dict with capture parameters
        """
        self.grid_map = grid_map
        self.config = config or {}
        self.rows = grid_map.get('rows', 5)
        self.cols = grid_map.get('cols', 10)
        self.mode_timeout_seconds = self.config.get('mode_timeout_seconds', 60)
        self.input_history = []

    def _validate_slot(self, row: int, col: int) -> bool:
        """Validate slot coordinates."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _parse_slot_string(self, slot_str: str) -> Optional[Tuple[int, int]]:
        """Parse single slot string like '2,3' or '2, 3'."""
        try:
            parts = slot_str.strip().replace(' ', '').split(',')
            if len(parts) != 2:
                return None
            row, col = int(parts[0]), int(parts[1])
            if self._validate_slot(row, col):
                return (row, col)
            return None
        except (ValueError, AttributeError):
            return None

    def _parse_multi_slot_string(self, slots_str: str) -> List[Tuple[int, int]]:
        """Parse multiple slot string like '1,2 3,4 5,6' or '1,2\\n3,4\\n5,6'."""
        slots = []
        # Split by space or newline
        slot_strs = slots_str.replace('\n', ' ').split()
        for slot_str in slot_strs:
            parsed = self._parse_slot_string(slot_str)
            if parsed:
                slots.append(parsed)
        return slots

    def _get_single_slot(self) -> Optional[Tuple[int, int]]:
        """Prompt user for single slot."""
        while True:
            user_input = input(f"Enter slot (row[0-{self.rows-1}],col[0-{self.cols-1}]) or [Q]uit: ").strip()
            self.input_history.append(user_input)
            
            if user_input.lower() in ['q', 'quit']:
                return None
            
            parsed = self._parse_slot_string(user_input)
            if parsed:
                return parsed
            
            print(f"  ✗ Invalid input. Expected format: '2,3' or '2, 3'")

    def _get_batch_slots(self, occupied_slots: set) -> Optional[List[Tuple[int, int]]]:
        """Prompt user for batch slots."""
        if not occupied_slots:
            print("  ✗ No occupied slots detected. Returning to mode selection.")
            return None
        
        occupied_sorted = sorted(list(occupied_slots))
        print(f"  Occupied slots: {', '.join([f'({r},{c})' for r, c in occupied_sorted])}")
        
        while True:
            user_input = input("Capture all? [Y]es / [N]o / [C]ustom / [Q]uit: ").strip().upper()
            self.input_history.append(user_input)
            
            if user_input == 'Q':
                return None
            elif user_input == 'Y':
                return occupied_sorted
            elif user_input == 'N':
                return None
            elif user_input == 'C':
                return self._get_custom_slots()
            else:
                print("  ✗ Invalid input. Enter Y, N, C, or Q.")

    def _get_custom_slots(self) -> Optional[List[Tuple[int, int]]]:
        """Prompt user for custom slot selection."""
        while True:
            user_input = input("Enter slots (e.g. '1,2 3,4 5,6') or [Q]uit: ").strip()
            self.input_history.append(user_input)
            
            if user_input.lower() == 'q':
                return None
            
            slots = self._parse_multi_slot_string(user_input)
            if slots:
                return slots
            
            print("  ✗ Invalid input. Use format: '1,2 3,4 5,6' or '1,2\\n3,4\\n5,6'")

    def get_capture_mode(self) -> Optional[CaptureMode]:
        """
        Prompt user for capture mode selection.

        Returns:
            CaptureMode.SINGLE or CaptureMode.BATCH, or None if quit
        """
        print("\n[CAPTURE MODE SELECTION]")
        while True:
            user_input = input("Select mode: [S]ingle / [B]atch / [Q]uit: ").strip().upper()
            self.input_history.append(user_input)
            
            if user_input == 'S':
                return CaptureMode.SINGLE
            elif user_input == 'B':
                return CaptureMode.BATCH
            elif user_input == 'Q':
                return None
            else:
                print("  ✗ Invalid input. Enter S, B, or Q.")

    def get_slots_to_capture(self, occupied_slots: Optional[set] = None) -> Optional[CaptureWorkflowResult]:
        """
        Get capture mode and slots to capture.

        Args:
            occupied_slots: Set of (row, col) tuples indicating occupied slots

        Returns:
            CaptureWorkflowResult or None if user quit
        """
        occupied_slots = occupied_slots or set()
        
        mode = self.get_capture_mode()
        if mode is None:
            return None
        
        if mode == CaptureMode.SINGLE:
            slot = self._get_single_slot()
            if slot is None:
                return self.get_slots_to_capture(occupied_slots)
            slots = [slot]
        else:  # BATCH
            slots = self._get_batch_slots(occupied_slots)
            if slots is None:
                return self.get_slots_to_capture(occupied_slots)
        
        # Sort slots by row, then col for consistent ordering
        slots = sorted(slots)
        
        return CaptureWorkflowResult(
            mode=mode,
            slots=slots,
            count=len(slots),
            user_input_history=self.input_history.copy(),
        )

    def confirm_capture(self, slot_id: Tuple[int, int]) -> bool:
        """
        Ask operator to confirm capture for specific slot.

        Args:
            slot_id: (row, col) tuple

        Returns:
            True to proceed, False to skip
        """
        user_input = input(f"Capture slot {slot_id}? [Y]es / [N]o / [S]kip all: ").strip().upper()
        self.input_history.append(user_input)
        return user_input == 'Y'

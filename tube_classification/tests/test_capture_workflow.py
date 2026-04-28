"""Tests for capture workflow state machine."""

import pytest
from tube_classification.src.preview.capture_workflow import CaptureWorkflow, CaptureMode


@pytest.fixture
def grid_map():
    return {
        'rows': 5,
        'cols': 10,
    }


@pytest.fixture
def workflow(grid_map):
    return CaptureWorkflow(grid_map)


def test_parse_single_slot(workflow):
    """Test parsing single slot string."""
    result = workflow._parse_slot_string("2,3")
    assert result == (2, 3)
    
    result = workflow._parse_slot_string("2, 3")
    assert result == (2, 3)
    
    result = workflow._parse_slot_string("invalid")
    assert result is None


def test_validate_slot(workflow):
    """Test slot validation."""
    assert workflow._validate_slot(0, 0) is True
    assert workflow._validate_slot(4, 9) is True
    assert workflow._validate_slot(5, 10) is False
    assert workflow._validate_slot(-1, 0) is False


def test_parse_multi_slot_string(workflow):
    """Test parsing multiple slots."""
    result = workflow._parse_multi_slot_string("1,2 3,4 5,6")
    assert result == [(1, 2), (3, 4), (5, 6)]
    
    # Test with newlines
    result = workflow._parse_multi_slot_string("1,2\n3,4\n5,6")
    assert result == [(1, 2), (3, 4), (5, 6)]


def test_invalid_slots_filtered(workflow):
    """Test that out-of-bounds slots are filtered."""
    result = workflow._parse_multi_slot_string("1,2 10,10 3,4")
    assert result == [(1, 2), (3, 4)]
    assert (10, 10) not in result


def test_mode_enum():
    """Test CaptureMode enum."""
    assert CaptureMode.SINGLE.value == "SINGLE"
    assert CaptureMode.BATCH.value == "BATCH"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

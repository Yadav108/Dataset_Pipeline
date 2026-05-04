# Test Suite Organization

This directory contains all project tests organized by category.

## Structure

```
tests/
├── README.md                    # This file
├── conftest.py                 # Pytest configuration
├── test_parser.py              # Config parser tests (unit)
├── test_capture_workflow.py    # Capture workflow tests (integration)
├── test_confirmation_preview.py # UI preview tests
├── test_live_preview.py        # Live preview tests
├── test_mode_routing.py        # Mode routing tests
├── legacy/                     # Legacy root-level test scripts
│   ├── test_verification.py    # Pipeline verification (old)
│   ├── test_startup.py         # Startup tests (old)
│   ├── test_preprocessing.py   # Preprocessing tests (old)
│   ├── test_complete_system.py # System tests (old)
│   ├── test_diagnostics.py     # Diagnostics (old)
│   └── ... (13 additional legacy test files)
└── hardware/                   # Hardware-dependent tests (skip in CI)
    └── (reserved for camera/device tests)
```

## Running Tests

### All tests
```bash
pytest
```

### Current test suite only (fast)
```bash
pytest tests/*.py -v
```

### Legacy tests (for verification/migration)
```bash
pytest tests/legacy/ -v
```

### Specific test category
```bash
pytest tests/test_parser.py -v          # Unit tests
pytest tests/test_capture_workflow.py -v # Integration tests
```

### With markers (future)
```bash
pytest -m "not hardware"  # Skip hardware tests in CI
pytest -m "unit"          # Only unit tests
```

## Test Categories

### Active Test Suite (`tests/*.py`)
- **test_parser.py** - Config parsing and validation (unit)
- **test_capture_workflow.py** - Capture workflow logic (integration)
- **test_confirmation_preview.py** - UI confirmation preview (integration)
- **test_live_preview.py** - Live preview rendering (integration)
- **test_mode_routing.py** - Mode routing logic (integration)

### Legacy Tests (`tests/legacy/`)
- Older ad-hoc test scripts from root directory
- Preserved for reference and gradual migration
- Can be run individually or as a group
- Candidates for refactoring or consolidation

### Hardware Tests (`tests/hardware/`)
- Tests requiring actual hardware (camera, etc.)
- Should have `@pytest.mark.hardware` decorator
- Skipped in CI environments

## Migration Notes

**Recent Changes:**
- Moved legacy test scripts from root to `tests/legacy/` (inactive storage)
- Active pytest suite remains in `tests/` root
- All tests remain runnable and unmodified

**For Developers:**
1. New tests should go in `tests/` (not root)
2. Use pytest conventions and fixtures
3. Mark hardware-dependent tests with `@pytest.mark.hardware`
4. Reference legacy tests for context but write to current standards

**For CI/CD:**
- Configure to run `pytest tests/*.py -v` (active suite)
- Mark hardware tests to skip in CI
- Legacy tests available for manual verification only

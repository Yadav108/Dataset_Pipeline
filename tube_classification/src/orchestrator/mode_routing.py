from typing import Any


def extract_bboxes_for_mode(
    capture_mode: str,
    roi_extractor: Any,
    depth_frame: Any,
) -> list[tuple[int, int, int, int]]:
    """Route ROI extraction to the correct extractor method by capture mode."""
    if capture_mode == "multi_top":
        return roi_extractor.extract_multi_top(depth_frame)
    if capture_mode == "single_top":
        bbox = roi_extractor.extract_top(depth_frame)
        return [bbox] if bbox is not None else []

    bbox = roi_extractor.extract(depth_frame)
    return [bbox] if bbox is not None else []

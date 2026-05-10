import numpy as np

from src.annotation import roi_extractor as roi_module
from src.annotation.roi_extractor import DepthROIExtractor


def test_multi_side_preserves_bottom_edge_components(monkeypatch):
    extractor = DepthROIExtractor()
    depth_frame = np.full((200, 200), 400, dtype=np.uint16)

    def fake_detect_camera_orientation(_depth_frame):
        return "side"

    def fake_preprocess(
        self,
        _depth_frame,
        _depth_min,
        _depth_max,
        initial_mask=None,
        remove_bottom_components=True,
    ):
        assert remove_bottom_components is False
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[60:160, 70:110] = 255
        return mask

    monkeypatch.setattr(roi_module, "detect_camera_orientation", fake_detect_camera_orientation)
    monkeypatch.setattr(DepthROIExtractor, "_preprocess_depth", fake_preprocess)
    monkeypatch.setattr(DepthROIExtractor, "_refine_with_candidate_depth", lambda self, **kwargs: kwargs["fallback_bbox"])
    monkeypatch.setattr(DepthROIExtractor, "_expand_single_side_bbox", lambda self, bbox, _shape: bbox)
    monkeypatch.setattr(DepthROIExtractor, "_find_tube_bottom_from_depth", lambda self, *_args, **_kwargs: 160)
    monkeypatch.setattr(DepthROIExtractor, "_is_holder_like_candidate", lambda self, *_args, **_kwargs: False)
    monkeypatch.setattr(DepthROIExtractor, "_is_tray_like", lambda self, *_args, **_kwargs: False)

    bboxes = extractor.extract_multi_side(depth_frame)

    assert bboxes == [(70, 60, 40, 100)]

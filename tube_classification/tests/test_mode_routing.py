from src.orchestrator.mode_routing import extract_bboxes_for_mode


class _DummyExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, depth_frame):
        self.calls.append(("extract", depth_frame))
        return (1, 2, 3, 4)

    def extract_top(self, depth_frame):
        self.calls.append(("extract_top", depth_frame))
        return (5, 6, 7, 8)

    def extract_multi_top(self, depth_frame):
        self.calls.append(("extract_multi_top", depth_frame))
        return [(9, 10, 11, 12), (13, 14, 15, 16)]


def test_single_side_uses_extract():
    depth_frame = object()
    extractor = _DummyExtractor()

    bboxes = extract_bboxes_for_mode("single_side", extractor, depth_frame)

    assert bboxes == [(1, 2, 3, 4)]
    assert extractor.calls == [("extract", depth_frame)]


def test_single_top_uses_extract_top():
    depth_frame = object()
    extractor = _DummyExtractor()

    bboxes = extract_bboxes_for_mode("single_top", extractor, depth_frame)

    assert bboxes == [(5, 6, 7, 8)]
    assert extractor.calls == [("extract_top", depth_frame)]


def test_multi_top_uses_extract_multi_top():
    depth_frame = object()
    extractor = _DummyExtractor()

    bboxes = extract_bboxes_for_mode("multi_top", extractor, depth_frame)

    assert bboxes == [(9, 10, 11, 12), (13, 14, 15, 16)]
    assert extractor.calls == [("extract_multi_top", depth_frame)]

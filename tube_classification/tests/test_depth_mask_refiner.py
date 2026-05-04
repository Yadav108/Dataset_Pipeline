import numpy as np

from src.processing.depth_mask_refiner import DepthMaskRefinerConfig, refine_mask


class _FakeIntrinsics:
    fy = 600.0


def test_refiner_removes_disconnected_zero_depth_holder_bleed():
    mask = np.zeros((480, 848), dtype=np.uint8)
    mask[80:260, 120:180] = 255
    mask[280:360, 80:260] = 255

    depth = np.full((480, 848), 380, dtype=np.uint16)
    depth[280:360, 80:260] = 0

    config = DepthMaskRefinerConfig(
        depth_band_tolerance_mm=35.0,
        zero_depth_support_kernel_size=9,
        apply_only_orientations=["side"],
    )

    refined, _ = refine_mask(
        mask=mask,
        depth_frame=depth,
        bbox=(80, 80, 180, 280),
        intrinsics=_FakeIntrinsics(),
        config=config,
        orientation="side",
    )

    assert np.count_nonzero(refined[280:360, 80:260]) == 0
    assert np.count_nonzero(refined[80:260, 120:180]) > 0


def test_refiner_prefers_tall_upper_tube_component_over_larger_lower_blob():
    mask = np.zeros((480, 848), dtype=np.uint8)
    mask[80:260, 140:200] = 255
    mask[280:360, 80:300] = 255

    depth = np.full((480, 848), 380, dtype=np.uint16)

    config = DepthMaskRefinerConfig(
        depth_band_tolerance_mm=45.0,
        zero_depth_support_kernel_size=9,
        apply_only_orientations=["side"],
    )

    refined, _ = refine_mask(
        mask=mask,
        depth_frame=depth,
        bbox=(80, 80, 220, 280),
        intrinsics=_FakeIntrinsics(),
        config=config,
        orientation="side",
    )

    assert np.count_nonzero(refined[80:260, 140:200]) > 0
    assert np.count_nonzero(refined[280:360, 80:300]) == 0


def test_refiner_zero_depth_fallback_preserves_small_tube_body():
    mask = np.zeros((480, 848), dtype=np.uint8)
    mask[90:250, 150:190] = 255

    depth = np.full((480, 848), 0, dtype=np.uint16)
    depth[90:118, 150:190] = 380

    config = DepthMaskRefinerConfig(
        depth_band_tolerance_mm=30.0,
        zero_depth_support_kernel_size=5,
        zero_depth_fallback_min_keep_ratio=0.55,
        zero_depth_fallback_min_zero_ratio=0.35,
        apply_only_orientations=["side"],
    )

    refined, _ = refine_mask(
        mask=mask,
        depth_frame=depth,
        bbox=(150, 90, 40, 180),
        intrinsics=_FakeIntrinsics(),
        config=config,
        orientation="side",
    )

    original_area = np.count_nonzero(mask)
    refined_area = np.count_nonzero(refined)
    assert refined_area >= int(0.70 * original_area)

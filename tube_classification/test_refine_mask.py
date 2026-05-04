import numpy as np
from src.processing.depth_mask_refiner import refine_mask, DepthMaskRefinerConfig

# Fake a 480x848 scene
mask = np.zeros((480, 848), dtype=np.uint8)
mask[80:260, 120:180] = 1          # tube body
mask[260:320, 100:200] = 1         # holder bleeding in

depth = np.full((480, 848), 380, dtype=np.uint16)
depth[260:320, :] = 480            # holder at different depth
depth[0:50, :] = 0                 # simulate failed measurements top rows

class FakeIntrinsics:
    fy = 600.0

config = DepthMaskRefinerConfig()

result = refine_mask(
    mask=mask,
    depth_frame=depth,
    bbox=(120, 80, 60, 240),
    intrinsics=FakeIntrinsics(),
    orientation="side",
    config=config,
)

print("Holder pixels remaining:", result[260:320, 120:180].sum())  # should be 0
print("Tube pixels remaining:", result[80:260, 120:180].sum())     # should be ~non-zero
print("Zero-depth region intact:", result[0:50, 120:180].sum())    # should be preserved

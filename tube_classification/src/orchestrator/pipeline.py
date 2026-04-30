import uuid
import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from config.parser import get_config
from src.acquisition.streamer import RealSenseStreamer
from src.acquisition.stability_detector import DepthStabilityDetector
from src.acquisition.distance_calibrator import DistanceCalibrator
from src.acquisition.volume_gate import run_volume_gate
from src.acquisition.capture_mode_gate import run_capture_mode_gate
from src.acquisition.pipeline_integration import PreprocessingPipeline
from src.annotation.roi_extractor import DepthROIExtractor
from src.annotation.sam_segmentor import SAMSegmentor
from src.annotation.metadata_builder import build_metadata
from src.annotation.annotation_writer import AnnotationWriter
from src.cleaning.blur_detector import BlurDetector
from src.cleaning.duplicate_remover import DuplicateRemover
from src.cleaning.bbox_quality_filter import BBoxQualityFilter
from src.cleaning.background_remover import BackgroundRemover
from src.orchestrator.mode_routing import extract_bboxes_for_mode
from src.export import ManifestBuilder, COCOExporter, YOLOExporter
from src.export.art_exporter import export_session_art_summary
from src.export.dataset_stats import (
    generate_balance_report,
    print_balance_report,
    print_balance_report_with_delta,
)


def _draw_preview_overlays(
    rgb_frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    consecutive_stable: int,
    stability_frames: int,
    current_depth_m: float,
    depth_min_m: float,
    depth_max_m: float,
    session_captures: int,
    target_count: int,
    last_rejection: str | None,
    rejection_frame_count: int,
) -> np.ndarray:
    """Draw live preview overlays on a copy of rgb_frame.
    
    Args:
        rgb_frame: Original RGB frame (not modified)
        bbox: ROI bounding box (x, y, w, h) or None
        consecutive_stable: Current consecutive stable frame count
        stability_frames: Target stability frame count
        current_depth_m: Current depth in meters
        depth_min_m: Minimum valid depth
        depth_max_m: Maximum valid depth
        session_captures: Images captured in this session
        target_count: Total target for class (existing + target for this session)
        last_rejection: Last rejection reason or None
        rejection_frame_count: Frames since last rejection (reset after 30)
        
    Returns:
        Frame with overlays drawn
    """
    # Use astype with copy=False to avoid unnecessary memory allocation
    # Only copy if needed for overlay drawing
    try:
        frame = rgb_frame.copy()
    except MemoryError:
        logger.warning("Memory low: Drawing overlays on original frame (may modify it)")
        frame = rgb_frame
    
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # 1. ROI bounding box (green if found, red if not)
    if bbox is not None:
        x, y, box_w, box_h = bbox
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
    
    # 2. Stability bar (top-left)
    is_stable = consecutive_stable >= stability_frames
    stability_color = (0, 255, 0) if is_stable else (0, 255, 255)  # Green or yellow
    stability_text = f"STABLE: {consecutive_stable}/{stability_frames}"
    cv2.putText(frame, stability_text, (10, 30), font, font_scale, stability_color, thickness)
    
    # 3. Depth zone status (below stability)
    in_depth_zone = depth_min_m <= current_depth_m <= depth_max_m
    depth_color = (0, 255, 0) if in_depth_zone else (0, 0, 255)  # Green or red
    depth_text = f"DEPTH: {current_depth_m:.3f}m  [{depth_min_m:.2f}–{depth_max_m:.2f}]"
    cv2.putText(frame, depth_text, (10, 50), font, font_scale, depth_color, thickness)
    
    # 4. Capture counter (bottom-left)
    capture_text = f"Captured: {session_captures} / {target_count}"
    cv2.putText(frame, capture_text, (10, h - 30), font, font_scale, (255, 255, 255), thickness)
    
    # 5. Last rejection reason (bottom-right, orange, clear after 30 frames)
    if last_rejection is not None and rejection_frame_count < 30:
        rejection_text = f"REJECTED: {last_rejection}"
        text_size = cv2.getTextSize(rejection_text, font, font_scale, thickness)[0]
        cv2.putText(
            frame,
            rejection_text,
            (w - text_size[0] - 10, h - 30),
            font,
            font_scale,
            (0, 165, 255),  # Orange (BGR)
            thickness,
        )
    
    return frame


def _process_roi(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    class_id: str,
    volume_ml: float,
    session_id: str,
    segmentor,
    writer,
    blur_detector,
    duplicate_remover,
    bbox_filter,
    background_remover,
    cfg,
    root: Path,
    image_id: str | None = None,
    preprocessing_pipeline: PreprocessingPipeline | None = None,
    calibration_result: dict | None = None,
    orientation: str = 'side',
    is_inverted: bool = False,
) -> tuple[bool, float | None]:
    """Process a single ROI through segmentation, annotation, and cleaning.
    
    Args:
        rgb_frame: RGB frame
        depth_frame: Depth frame
        bbox: ROI bounding box (x, y, w, h)
        class_id, volume_ml, session_id: Metadata
        segmentor, writer, cleaners: Processing components
        cfg: Config
        root: Storage root path
        image_id: Optional image_id; if None, generates one
        preprocessing_pipeline: Optional preprocessing pipeline (NEW)
        calibration_result: Distance calibration result dict (NEW)
        
    Returns:
        Tuple of (success: bool, iou_score: float | None).
        success: True if successfully captured and saved
        iou_score: SAM IoU score (0.0-1.0) if successful, None otherwise
    """
    if image_id is None:
        image_id = f"{class_id}_{uuid.uuid4().hex[:8]}"
    
    quality_metrics = None

    # NEW: Preprocess depth if pipeline provided
    if preprocessing_pipeline:
        depth_frame, quality_metrics, preprocess_stats = preprocessing_pipeline.process_depth_frame(
            depth_frame=depth_frame,
            rgb_frame=rgb_frame,
            frame_id=image_id,
            compute_metrics=True
        )
        
        # Log preprocessing stats
        if quality_metrics:
            logger.debug(
                f"Preprocessing: quality={quality_metrics.quality_score:.2f}, "
                f"time={preprocess_stats['total_time_ms']:.1f}ms"
            )
    
    # Segment with SAM — pass depth_frame for holder-suppression depth filter
    segment_result = segmentor.segment(rgb_frame, bbox, depth_frame=depth_frame, orientation=orientation, is_inverted=is_inverted)
    if segment_result is None:
        return (False, None)
    
    mask, sam_iou_score = segment_result
    
    # NEW: Refine mask with depth geometry if available
    if preprocessing_pipeline and preprocessing_pipeline.mask_refinement_enabled:
        original_mask = mask.copy()
        refined_mask, refinement_stats = preprocessing_pipeline.refine_mask(
            depth_frame=depth_frame,
            mask=mask,
            frame_id=image_id
        )
        if refinement_stats.get('refined', False):
            orig_area = float(np.count_nonzero(original_mask))
            ref_area = float(np.count_nonzero(refined_mask))
            area_ratio = ref_area / max(orig_area, 1.0)

            orig_rows = np.where(original_mask.any(axis=1))[0]
            ref_rows = np.where(refined_mask.any(axis=1))[0]
            orig_cols = np.where(original_mask.any(axis=0))[0]
            ref_cols = np.where(refined_mask.any(axis=0))[0]

            orig_h = int(orig_rows[-1] - orig_rows[0] + 1) if len(orig_rows) > 0 else 0
            ref_h = int(ref_rows[-1] - ref_rows[0] + 1) if len(ref_rows) > 0 else 0
            orig_w = int(orig_cols[-1] - orig_cols[0] + 1) if len(orig_cols) > 0 else 0
            ref_w = int(ref_cols[-1] - ref_cols[0] + 1) if len(ref_cols) > 0 else 0

            # Guard against refinement collapsing masks into small/diagonal fragments.
            bad_shrink = area_ratio < 0.75
            bad_height = (orig_h >= 60) and (ref_h < int(0.70 * orig_h))
            bad_width = (orig_w >= 35) and (ref_w < int(0.55 * orig_w))

            if bad_shrink or bad_height or bad_width:
                logger.debug(
                    f"Mask refinement rejected: area_ratio={area_ratio:.2f}, "
                    f"height={orig_h}->{ref_h}, width={orig_w}->{ref_w}"
                )
            else:
                mask = refined_mask
                logger.debug(
                    f"Mask refinement accepted: IoU+{refinement_stats['iou_improvement_pct']:.1f}%, "
                    f"correlation={refinement_stats['depth_correlation']:.2f}, "
                    f"area_ratio={area_ratio:.2f}"
                )
    
    logger.debug(f"Frame SAM segmentation successful: {image_id} (IoU={sam_iou_score:.4f})")
    
    # Build and write annotation (pass IoU score and calibration)
    calibration_metadata = None
    if calibration_result:
        calibration_metadata = {
            'distance_calibration': {
                'measured_distance_m': calibration_result.get('measured_distance_m'),
                'calibrated_depth_min_m': calibration_result.get('calibrated_depth_min_m'),
                'calibrated_depth_max_m': calibration_result.get('calibrated_depth_max_m'),
            }
        }
    
    metadata = build_metadata(
        image_id,
        class_id,
        volume_ml,
        bbox,
        mask,
        rgb_frame.shape,
        sam_iou_score=sam_iou_score,
        calibration_metadata=calibration_metadata,
    )
    if quality_metrics is not None:
        metadata["quality_metrics"] = {
            "image_id": image_id,
            "class_id": class_id,
            "blur_score": float(getattr(quality_metrics, "blur_score", 0.0)),
            "coverage_ratio": float(getattr(quality_metrics, "coverage_ratio", 0.0)),
            "sam_iou_score": float(sam_iou_score if sam_iou_score is not None else 0.0),
            "depth_variance": float(getattr(quality_metrics, "depth_variance", 0.0)),
            "quality_score": float(getattr(quality_metrics, "quality_score", 0.0)),
        }
    
    raw_dir = root / "raw" / class_id / session_id
    ann_dir = root / "annotations" / class_id / session_id
    
    writer.write(
        image_id,
        class_id,
        session_id,
        rgb_frame,
        depth_frame,
        mask,
        bbox,
        metadata,
    )
    
    # Run cleaning pipeline
    cleaned_raw = root / "cleaned" / "raw" / class_id / session_id
    cleaned_ann = root / "cleaned" / "annotations" / class_id / session_id
    
    blur_detector.filter_directory(raw_dir, cleaned_raw)
    duplicate_remover.remove_duplicates(cleaned_raw)
    bbox_filter.filter_directory(raw_dir, ann_dir, cleaned_raw, cleaned_ann)
    
    # Background removal
    if cfg.pipeline.background_removal:
        try:
            x, y, box_w, box_h = bbox
            # Use a moderate expansion so rembg sees full tube body even if ROI is tight.
            pad_x = max(20, int(round(box_w * 0.40)))
            pad_y = max(24, int(round(box_h * 0.35)))
            crop_x0 = max(0, x - pad_x)
            crop_y0 = max(0, y - pad_y)
            crop_x1 = min(rgb_frame.shape[1], x + box_w + pad_x)
            crop_y1 = min(rgb_frame.shape[0], y + box_h + pad_y)

            roi_crop = rgb_frame[crop_y0:crop_y1, crop_x0:crop_x1]
            nobg_crop = background_remover.remove_from_array(roi_crop)

            # rembg-first path: use rembg alpha and only prune disconnected noise.
            if nobg_crop.ndim == 3 and nobg_crop.shape[2] == 4:
                rgba = nobg_crop.copy()
            elif nobg_crop.ndim == 3 and nobg_crop.shape[2] == 3:
                rgba = cv2.cvtColor(nobg_crop, cv2.COLOR_BGR2BGRA)
            else:
                raise ValueError(
                    f"Unexpected background remover output shape: {nobg_crop.shape}"
                )

            rembg_alpha = rgba[:, :, 3]
            bbox_area = float(max(1, (crop_x1 - crop_x0) * (crop_y1 - crop_y0)))
            rembg_binary = (rembg_alpha > 0).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                rembg_binary,
                connectivity=8,
            )
            rembg_component_alpha = (rembg_binary * 255).astype(np.uint8)
            rembg_best_area = float(np.count_nonzero(rembg_component_alpha))

            if num_labels > 2:
                roi_cx = rembg_component_alpha.shape[1] * 0.5
                best_label = -1
                best_score = -1e9

                for lbl in range(1, num_labels):
                    area = float(stats[lbl, cv2.CC_STAT_AREA])
                    if area < 40.0:
                        continue

                    left = float(stats[lbl, cv2.CC_STAT_LEFT])
                    width = float(max(1, stats[lbl, cv2.CC_STAT_WIDTH]))
                    height = float(max(1, stats[lbl, cv2.CC_STAT_HEIGHT]))
                    cx = left + (width * 0.5)
                    height_cov = min(height / max(float(rembg_component_alpha.shape[0]), 1.0), 1.4)
                    area_cov = min(area / bbox_area, 1.6)
                    center_score = max(
                        0.0,
                        1.0 - (abs(cx - roi_cx) / max(float(rembg_component_alpha.shape[1]) * 0.70, 1.0)),
                    )

                    score = (
                        0.42 * area_cov
                        + 0.33 * height_cov
                        + 0.25 * center_score
                    )
                    if score > best_score:
                        best_score = score
                        best_label = lbl
                        rembg_best_area = area

                if best_label > 0:
                    rembg_component_alpha = np.zeros_like(rembg_alpha, dtype=np.uint8)
                    rembg_component_alpha[labels == best_label] = 255

            final_alpha = rembg_component_alpha
            # Safety: never output near-empty masks; fall back to raw rembg alpha.
            if rembg_best_area < max(120.0, 0.05 * bbox_area):
                final_alpha = (rembg_binary * 255).astype(np.uint8)

            rgba[:, :, 3] = final_alpha
            nobg_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
             
            nobg_path = ann_dir / f"{image_id}_rgb_nobg.png"
            cv2.imwrite(str(nobg_path), nobg_bgr)
            logger.debug(f"Background removed: {image_id}")
        except Exception as e:
            logger.warning(f"Background removal failed for {image_id}: {e}")
    
    return (True, sam_iou_score)


def _select_tube_class(matched_tubes: list[dict]) -> str:
    """Interactively select or input tube class for capturing.
    
    Args:
        matched_tubes: List of matched tube classes from volume gate
        
    Returns:
        Selected tube class_id (string)
    """
    print("\n" + "="*60)
    print("TUBE CLASS SELECTION")
    print("="*60)
    
    # If only one tube class matched, use it
    if len(matched_tubes) == 1:
        class_id = matched_tubes[0]["class_id"]
        print(f"✓ Single tube class detected: {class_id}")
        return class_id
    
    # Multiple classes: present selection or allow custom input
    print(f"Found {len(matched_tubes)} tube classes from volume detection:")
    for i, tube in enumerate(matched_tubes):
        print(f"  [{i}] {tube['class_id']}")
    
    print("\nOptions:")
    print("  - Enter number (0-" + str(len(matched_tubes)-1) + ") to select from above")
    print("  - Enter custom class name (e.g., SAR_RED_MEDIUM)")
    
    while True:
        user_input = input("\nTube class: ").strip()
        
        # Try to parse as index
        if user_input.isdigit():
            choice = int(user_input)
            if 0 <= choice < len(matched_tubes):
                class_id = matched_tubes[choice]["class_id"]
                print(f"✓ Selected: {class_id}")
                return class_id
            else:
                print(f"✗ Invalid index. Enter 0-{len(matched_tubes)-1}")
                continue
        
        # Custom class name
        if user_input and len(user_input) >= 3:
            # Validate: alphanumeric + underscore only
            if all(c.isalnum() or c == '_' for c in user_input):
                class_id = user_input.upper()
                confirm = input(f"Confirm custom class: {class_id}? [y/N]: ").strip().lower()
                if confirm == 'y':
                    print(f"✓ Using custom class: {class_id}")
                    return class_id
                else:
                    continue
            else:
                print("✗ Invalid characters. Use alphanumeric and underscore only.")
                continue
        
        print("✗ Invalid input. Try again.")


def run_pipeline(
    preselected_volume_ml: float | None = None,
    preselected_class_id: str | None = None,
    preselected_capture_mode: str | None = None,
) -> None:
    """Run automated capture pipeline.
    
    Orchestrates volume declaration, session setup, component initialization,
    and the main capture loop with integrated cleaning filters.
    """
    
    # SETUP
    cfg = get_config()
    if preselected_volume_ml is None:
        volume_ml, matched_tubes = run_volume_gate()
    else:
        volume_ml = preselected_volume_ml
        matched_tubes = [{"class_id": preselected_class_id}] if preselected_class_id else []

    capture_mode = preselected_capture_mode or run_capture_mode_gate()
    
    session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Handle class selection — NEW: Interactive prompt
    class_id = preselected_class_id or _select_tube_class(matched_tubes)
    
    logger.info(
        f"Session started: {session_id} | class={class_id} | volume={volume_ml}ml"
    )
    
    # COUNT EXISTING IMAGES FOR THIS CLASS (crash recovery)
    root = Path(cfg.storage.root_dir)
    class_raw_dir = root / "raw" / class_id
    existing_count = 0
    if class_raw_dir.exists():
        # Count all _rgb.png files across all prior sessions
        existing_count = len(list(class_raw_dir.rglob("*_rgb.png")))
    
    target = cfg.pipeline.target_images_per_class
    remaining = target - existing_count
    
    logger.info(
        f"Class {class_id}: {existing_count} existing / {target} target. "
        f"Need {remaining} more this session."
    )
    
    # Check if target already met
    if existing_count >= target:
        logger.warning(
            f"WARNING: Class {class_id} already has {existing_count} images (target met). "
            f"Continue anyway? [y/N]"
        )
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            logger.info("Stopping without capturing. Exiting.")
            return
    
    # PRINT DATASET BALANCE REPORT AT STARTUP
    startup_report = generate_balance_report()
    print_balance_report(startup_report, title="DATASET BALANCE AT SESSION START")
    
    # INITIALIZE COMPONENTS
    streamer = RealSenseStreamer()
    streamer.start()  # Start first to get depth_scale
    
    # Pass actual depth scale to components that need it
    detector = DepthStabilityDetector(depth_scale=streamer.depth_scale)
    roi_extractor = DepthROIExtractor(depth_scale=streamer.depth_scale)
    segmentor = SAMSegmentor()
    segmentor.load()
    writer = AnnotationWriter()
    blur_detector = BlurDetector()
    duplicate_remover = DuplicateRemover()
    bbox_filter = BBoxQualityFilter()
    background_remover = BackgroundRemover()
    
    # NEW: Preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    logger.info(
        f"Preprocessing pipeline initialized: "
        f"bilateral={preprocessing_pipeline.bilateral_enabled}, "
        f"temporal_smoothing={preprocessing_pipeline.temporal_smoothing_enabled}, "
        f"inpainting={preprocessing_pipeline.inpainting_enabled}, "
        f"mask_refinement={preprocessing_pipeline.mask_refinement_enabled}"
    )
    
    # NEW: Auto-calibrate camera-to-tube distance (DISABLED FOR STABILITY)
    # Distance calibration was causing overly restrictive depth ranges
    # Reverting to config-based depth settings for reliability
    calibration_result = None
    logger.info("Distance calibration: DISABLED (using config defaults for stability)")
    
    # STREAM + CAPTURE LOOP
    logger.info("Streaming started. Place tube in capture zone.")
    
    captured_count = 0
    session_captures = 0  # Track captures THIS SESSION only (for target check)
    frame_count = 0
    rejected_stability = 0
    rejected_roi = 0
    rejected_seg = 0
    stability_achieved = False  # Track if we've ever seen stable frames
    last_known_roi = None  # Store last successful ROI for  interference mitigation
    last_rejection = None  # Track rejection reason for preview
    rejection_frame_count = 31  # Start > 30 to not display initially
    
    # Quality tracking for R2.4
    total_annotation_attempts = 0
    high_quality_annotations = 0
    
    try:
        while True:
            # Get aligned frames
            frames = streamer.get_aligned_frames()
            if frames is None:
                continue
            
            frame_count += 1
            rgb_frame, depth_frame = frames
            
            # Get depth info for preview (convert roi center to meters)
            depth_m = depth_frame.astype(np.float32) * detector.depth_scale
            center_depth = np.median(depth_m[depth_m > 0]) if np.any(depth_m > 0) else 0.0
            
            # Select mode-specific depth range
            if capture_mode == "single_top":
                depth_min = cfg.pipeline.top_depth_min_m
                depth_max = cfg.pipeline.top_depth_max_m
            else:
                depth_min = cfg.camera.depth_min_m
                depth_max = cfg.camera.depth_max_m
            
            # SKIP STABILITY CHECK - capture immediately
            logger.debug(f"Frame {frame_count}: Skipped stability check, proceeding to ROI extraction")
            
            # Extract ROI(s) — mode-dependent
            bboxes = extract_bboxes_for_mode(capture_mode, roi_extractor, depth_frame)
            frame_orientation = roi_extractor.orientation
            frame_is_inverted = roi_extractor.is_inverted
            logger.debug(
                f"Frame {frame_count}: orientation={frame_orientation}, "
                f"is_inverted={frame_is_inverted}"
            )

            if not bboxes:
                rejected_roi += 1
                last_rejection = "NO_ROI"
                rejection_frame_count = 0
                # Stability lost - reset
                stability_achieved = False
                detector.reset()
                
                # Draw and display preview even on rejection
                if cfg.pipeline.show_preview:
                    preview = _draw_preview_overlays(
                        rgb_frame, None, detector.consecutive_stable,
                        cfg.pipeline.stability_frames, center_depth, depth_min, depth_max,
                        session_captures, existing_count + session_captures + remaining,
                        last_rejection, rejection_frame_count,
                    )
                    cv2.imshow("Tube Classification — Live Preview", preview)
                    cv2.waitKey(1)
                
                rejection_frame_count += 1
                continue
            
            # Store first ROI for next frame's stability check (mitigates hand interference)
            last_known_roi = bboxes[0] if bboxes else None
            
            logger.debug(f"Frame {frame_count}: Extracted {len(bboxes)} ROI(s)")
            
            # Process each ROI independently
            root = Path(cfg.storage.root_dir)
            rois_captured_this_frame = 0
            first_bbox_for_preview = bboxes[0]  # For preview overlay
            
            for bbox in bboxes:
                success, iou_score = _process_roi(
                    rgb_frame, depth_frame, bbox,
                    class_id, volume_ml, session_id,
                    segmentor, writer, blur_detector,
                    duplicate_remover, bbox_filter, background_remover,
                    cfg, root,
                    preprocessing_pipeline=preprocessing_pipeline,
                    calibration_result=calibration_result,
                    orientation=frame_orientation,
                    is_inverted=frame_is_inverted,
                )
                
                if success:
                    rois_captured_this_frame += 1
                    total_annotation_attempts += 1
                    # Track high-quality annotations (IoU >= threshold)
                    if iou_score is not None and iou_score >= cfg.pipeline.sam_iou_threshold:
                        high_quality_annotations += 1
            
            # Track rejections
            if rois_captured_this_frame == 0:
                # All ROIs failed SAM segmentation
                rejected_seg += len(bboxes)
                # Count failed attempts too
                total_annotation_attempts += len(bboxes)
                last_rejection = "SAM_FAIL"
                rejection_frame_count = 0
                # Stability lost - reset
                stability_achieved = False
                detector.reset()
                
                # Draw and display preview even on rejection
                if cfg.pipeline.show_preview:
                    preview = _draw_preview_overlays(
                        rgb_frame, first_bbox_for_preview, detector.consecutive_stable,
                        cfg.pipeline.stability_frames, center_depth, depth_min, depth_max,
                        session_captures, existing_count + session_captures + remaining,
                        last_rejection, rejection_frame_count,
                    )
                    cv2.imshow("Tube Classification — Live Preview", preview)
                    cv2.waitKey(1)
                
                rejection_frame_count += 1
                continue
            
            # Update counters - DO NOT RESET detector (continue capturing)
            captured_count += rois_captured_this_frame
            session_captures += rois_captured_this_frame  # Track session captures separately for target check
            last_rejection = None  # Clear rejection after successful capture
            rejection_frame_count = 31
            logger.info(f"Captured {captured_count} images ({rois_captured_this_frame} ROIs this frame). Continuing capture session...")
            
            # Draw and display preview on successful capture
            if cfg.pipeline.show_preview:
                preview = _draw_preview_overlays(
                    rgb_frame, first_bbox_for_preview, detector.consecutive_stable,
                    cfg.pipeline.stability_frames, center_depth, depth_min, depth_max,
                    session_captures, existing_count + session_captures,
                    last_rejection, rejection_frame_count,
                )
                cv2.imshow("Tube Classification — Live Preview", preview)
                cv2.waitKey(1)
            
            # Periodic checkpoint: save manifest snapshot for crash recovery
            if captured_count % cfg.pipeline.checkpoint_interval == 0:
                ManifestBuilder().build()
                logger.info(f"Checkpoint saved — {captured_count} captures completed.")
            
            # Explicit cleanup to prevent memory leaks
            del rgb_frame, depth_frame, depth_m
            
            # Check if target reached for this class
            if existing_count + session_captures >= target:
                logger.info(
                    f"Target reached for class {class_id}. "
                    f"({existing_count + session_captures}/{target} images collected). Stopping capture."
                )
                raise KeyboardInterrupt  # Trigger clean session end
    
    except KeyboardInterrupt:
        streamer.stop()
        
        # NEW: Print preprocessing pipeline statistics
        preprocess_stats = preprocessing_pipeline.get_statistics()
        
        # Print diagnostics
        logger.info(f"Pipeline stopped.")
        logger.info(f"\n{'='*70}")
        logger.info(f"CAPTURE SESSION DIAGNOSTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Total frames processed:     {frame_count}")
        logger.info(f"Frames captured:            {captured_count}")
        logger.info(f"Rejected (stability):       {rejected_stability}")
        logger.info(f"Rejected (no ROI):          {rejected_roi}")
        logger.info(f"Rejected (SAM failed):      {rejected_seg}")
        
        if frame_count > 0:
            capture_rate = (captured_count / frame_count) * 100
            logger.info(f"\nCapture rate:               {capture_rate:.1f}%")
        
        # NEW: Log preprocessing statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"PREPROCESSING PIPELINE STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Mode:                       {preprocess_stats['mode']}")
        logger.info(f"Frames preprocessed:        {preprocess_stats['total_frames']}")
        logger.info(f"Avg time per frame:         {preprocess_stats['avg_time_per_frame_ms']:.1f}ms")
        logger.info(f"Total preprocessing time:   {preprocess_stats['total_time_ms']/1000:.1f}s")
        
        enabled_steps = [k for k, v in preprocess_stats['enabled_steps'].items() if v]
        logger.info(f"Enabled steps:              {', '.join(enabled_steps)}")
        
        logger.info(f"{'='*70}\n")
        
        # Annotation quality tracking (R2.4)
        if total_annotation_attempts > 0:
            annotation_accuracy = (high_quality_annotations / total_annotation_attempts) * 100
            logger.info(
                f"Annotation accuracy this session: {annotation_accuracy:.1f}% "
                f"({high_quality_annotations}/{total_annotation_attempts} frames met IoU ≥ {cfg.pipeline.sam_iou_threshold})"
            )
            if annotation_accuracy < 95.0:
                logger.warning(f"R2.4 threshold not met — annotation accuracy below 95%")
        else:
            logger.warning("No annotations attempted this session.")
        
        logger.info(f"Total captured this session: {captured_count}")
        
        # PRINT DATASET BALANCE REPORT AT SESSION END
        final_report = generate_balance_report()
        print_balance_report_with_delta(
            final_report,
            previous_report=startup_report,
            title="DATASET BALANCE AT SESSION END"
        )
        
        logger.info("Building dataset manifest...")
        ManifestBuilder().build()

        logger.info("Exporting to COCO format...")
        COCOExporter().export()

        logger.info("Exporting to YOLO format...")
        YOLOExporter().export()

        # Export Quality Constellation visualization
        if cfg.pipeline.enable_art_export:
            try:
                logger.info("Generating Quality Constellation visualization...")
                # Load metrics from written annotation metadata.
                metrics_list = []
                import json
                from types import SimpleNamespace
                metadata_root = Path(cfg.storage.root_dir) / "annotations"
                for metadata_path in metadata_root.rglob("*_metadata.json"):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    qm = metadata.get("quality_metrics")
                    if qm:
                        metrics_list.append(SimpleNamespace(**qm))
                
                if metrics_list:
                    export_session_art_summary(
                        session_id=session_id,
                        metrics_list=metrics_list,
                        output_dir=Path(cfg.storage.root_dir)
                    )
                    logger.info(f"Quality Constellation saved to dataset/art/{session_id}/")
                else:
                    logger.warning("No quality metrics found for constellation generation")
            except Exception as e:
                logger.error(f"Failed to generate Quality Constellation: {e}")

        logger.info("=== Pipeline complete. Dataset ready ===")
        
        # Cleanup
        cv2.destroyAllWindows()

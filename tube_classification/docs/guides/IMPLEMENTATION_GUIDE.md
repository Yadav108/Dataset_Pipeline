# Image Quality Optimization - Implementation Guide

## Quick Start

### 1. Analyze Current Quality
Run analysis on your existing dataset to understand current performance:
```bash
cd C:\Users\Aryan\OneDrive\Desktop\Projects\Dataset_Pipeline\tube_classification
python analyze_image_quality.py
```

This will generate a detailed quality report showing:
- Blur score distribution (Laplacian variance)
- Coverage ratio distribution (mask/bbox overlap)
- SAM IoU score distribution (segmentation confidence)
- Rejection breakdown by reason

**Output:**
- Console report with statistics
- CSV file: `quality_metrics.csv` with detailed per-image metrics

### 2. Generate Recommendations
Run the optimizer to get tuning recommendations:
```bash
python optimize_config.py > recommendations.json
```

**Output:** Recommended threshold values with rationale

---

## Configuration Options

### Option A: Conservative Tuning (Recommended - Balanced)
**Best for:** Improving dataset yield while maintaining quality

**Changes to `config/config.yaml`:**
```yaml
pipeline:
  blur_threshold: 50.0              # from 40.0 (less strict)
  min_coverage_ratio: 0.45          # from 0.35 (stricter)
  
confirmation:
  blur_threshold: 85.0              # from 100.0 (less strict for operators)
  mask_confidence_threshold: 0.75   # from 0.85 (less strict)
```

**Expected Results:**
- ✓ 20-25% fewer blur rejections
- ✓ Better mask quality consistency
- ✓ Overall dataset yield: +15-20%
- ✓ Operator review time: unchanged

### Option B: Maximum Quality Tuning
**Best for:** Critical applications requiring highest quality

**Changes to `config/config.yaml`:**
```yaml
pipeline:
  blur_threshold: 55.0              # stricter
  min_coverage_ratio: 0.50          # much stricter
  sam_iou_threshold: 0.65           # stricter

confirmation:
  blur_threshold: 80.0              # stricter
  mask_confidence_threshold: 0.80   # stricter
```

**Expected Results:**
- ✓ Highest quality dataset (~98% acceptable)
- ✗ ~30-40% fewer images pass filters
- ✓ Less operator review needed (fewer edge cases)

### Option C: Production Balance
**Best for:** Large-scale capture with good quality/yield balance

**Changes to `config/config.yaml`:**
```yaml
pipeline:
  blur_threshold: 48.0              # moderate
  min_coverage_ratio: 0.45          # moderate
  sam_iou_threshold: 0.58           # slightly relaxed

confirmation:
  blur_threshold: 75.0              # moderate
  mask_confidence_threshold: 0.70   # relaxed
  coverage_ratio_min: 0.40          # operator reference
```

---

## Implementation Steps

### Step 1: Backup Current Configuration
```bash
# Backup original config
copy config\config.yaml config\config_backup_$(date +%Y%m%d).yaml
```

### Step 2: Apply Configuration Changes
**Option A: Manual Edit**
1. Open `config/config.yaml` in your editor
2. Update the following values:
   - `blur_threshold: 50.0` (line 19)
   - `min_coverage_ratio: 0.45` (line 21)
   - `confirmation.blur_threshold: 85.0` (line 73)
   - `confirmation.mask_confidence_threshold: 0.75` (line 74)

**Option B: Use Provided Config**
```bash
# Copy the optimized config
copy config\config_optimized.yaml config\config.yaml
```

### Step 3: Test on Small Batch
Create a test capture session with 50-100 images:
```bash
python capture.py
# Select single volume class
# Capture 50 images
# Review acceptance/rejection rates
```

### Step 4: Compare Results
Analyze the test batch:
```bash
python analyze_image_quality.py
```

Compare metrics:
- Old blur threshold: % rejected
- New blur threshold: % rejected
- Coverage ratio improvements
- Overall yield improvement

### Step 5: Validate Quality
- [ ] Manually review 20-30 "borderline" auto-accepted images
- [ ] Check for obvious blurry images that should be rejected
- [ ] Verify operator feedback aligns with auto-decisions
- [ ] Confirm no loss in dataset quality despite higher yield

### Step 6: Deploy
If validation successful:
```bash
# Commit changes
git add config/config.yaml
git commit -m "Optimize image quality thresholds

- blur_threshold: 40.0 → 50.0 (reduce false rejects)
- min_coverage_ratio: 0.35 → 0.45 (improve mask quality)
- confirmation.blur_threshold: 100.0 → 85.0 (better operator consistency)

Expected: +15-20% yield, no quality regression

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Monitoring Quality

### Weekly Quality Report
After each capture session, generate a quality summary:

```bash
# Analyze session results
python analyze_image_quality.py

# Compare to baseline
# - Track rejection rates over time
# - Monitor coverage ratio trends
# - Watch for SAM confidence changes
```

### Key Metrics to Track
1. **Yield Rate**: % images passing all filters
   - Target: 75-85% with optimized config
   - Track trend over time

2. **Rejection Breakdown**:
   - Blur: should be ~15-20% of rejections
   - Coverage: should be ~30-40% of rejections
   - Duplicates: should be ~10-15% of rejections
   - SAM IoU: should be ~20-30% of rejections

3. **Quality Scores**:
   - Median blur score: should be > 48
   - Median coverage: should be > 0.45
   - Median SAM IoU: should be > 0.65

4. **Operator Feedback**:
   - % auto-accepted but manually rejected: should be < 5%
   - % auto-rejected but operator wanted: should be < 5%

---

## Troubleshooting

### Problem: Too many images still rejected
**Solution:**
1. Run analysis to check blur distribution
2. If median < 45, further increase blur_threshold
3. Check coverage_ratio - if median < 0.40, coverage threshold too strict
4. Review SAM IoU distribution - may need to relax sam_iou_threshold

### Problem: Quality decreasing despite higher yield
**Solution:**
1. Check operator feedback - are rejections increasing?
2. Run analysis to compare to baseline
3. If blur rejections decrease but others increase → other thresholds too permissive
4. If coverage ratio images look bad → increase coverage_threshold
5. If segmentation looks poor → increase sam_iou_threshold

### Problem: Operator consistently rejecting auto-accepted images
**Solution:**
1. Make confirmation preview stricter (increase blur_threshold)
2. Add visual indicator of why image was accepted
3. Enable quality metrics display in confirmation window
4. Let operator override individual images without affecting main pipeline

---

## Rollback

If new configuration causes issues:
```bash
# Restore previous config
copy config\config_backup_YYYYMMDD.yaml config\config.yaml

# Or use git
git checkout config\config.yaml

# Restart capture with original thresholds
python capture.py
```

---

## Documentation Files

This implementation includes:

1. **IMAGE_QUALITY_OPTIMIZATION.md** - Detailed analysis and recommendations
2. **analyze_image_quality.py** - Quality analysis tool
3. **optimize_config.py** - Configuration recommendation engine
4. **config_optimized.yaml** - Recommended configuration

---

## Support & Questions

For questions about:
- **Blur threshold tuning**: See Section 4.1 in IMAGE_QUALITY_OPTIMIZATION.md
- **Coverage ratio tuning**: See Section 4.2 in IMAGE_QUALITY_OPTIMIZATION.md
- **SAM confidence**: See Section 4.3 in IMAGE_QUALITY_OPTIMIZATION.md
- **Implementation issues**: Follow troubleshooting guide above


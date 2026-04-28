#!/usr/bin/env python3
"""
Quality Configuration Optimizer - Generate optimized config based on analysis
Recommends threshold adjustments to improve dataset quality
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdRecommendation:
    """Recommendation for a configuration threshold."""
    parameter: str
    current_value: float
    recommended_value: float
    rationale: str
    impact: str  # "high", "medium", "low"


class ConfigOptimizer:
    """Generate configuration recommendations based on analysis results."""
    
    @staticmethod
    def recommend_blur_threshold(report: dict) -> ThresholdRecommendation:
        """Recommend blur threshold based on distribution."""
        blur_stats = report.get("blur_analysis", {})
        current_threshold = blur_stats.get("threshold", 40.0)
        
        # Analyze rejection rate
        sharp_pct = blur_stats.get("sharp_percentage", 0)
        p25 = blur_stats.get("distribution_percentiles", {}).get("p25", current_threshold)
        p50 = blur_stats.get("distribution_percentiles", {}).get("p50", current_threshold)
        
        # If >50% images are being rejected, threshold is too aggressive
        if sharp_pct < 50:
            recommended = p50 - 10  # More permissive
            rationale = f"Current threshold rejects {100-sharp_pct:.1f}% of images. Setting to p50-10 allows more valid captures."
        # If <10% rejection, threshold could be stricter
        elif sharp_pct > 95:
            recommended = p75 + 5  # More strict
            rationale = f"Very few rejections suggest threshold is too permissive. Tightening to p75+5."
        # Otherwise, use median-based threshold
        else:
            recommended = p50
            rationale = f"Current threshold is reasonable. Using median ({p50:.1f}) for optimal balance."
        
        recommended = max(30, min(recommended, 100))  # Clamp to sensible range
        
        return ThresholdRecommendation(
            parameter="blur_threshold",
            current_value=current_threshold,
            recommended_value=recommended,
            rationale=rationale,
            impact="high"
        )
    
    @staticmethod
    def recommend_coverage_threshold(report: dict) -> ThresholdRecommendation:
        """Recommend coverage ratio threshold based on distribution."""
        cov_stats = report.get("coverage_analysis", {})
        current_threshold = cov_stats.get("threshold", 0.35)
        good_pct = cov_stats.get("good_coverage_percentage", 0)
        mean_coverage = cov_stats.get("mean", 0)
        median_coverage = cov_stats.get("median", 0)
        
        # If >70% pass, threshold is too permissive
        if good_pct > 70:
            recommended = min(0.50, median_coverage - 0.10)  # Increase threshold
            rationale = f"{good_pct:.1f}% pass current threshold. Recommending stricter {recommended:.2f} (median - 0.10)."
        # If <40% pass, threshold is too strict
        elif good_pct < 40:
            recommended = max(0.25, median_coverage - 0.05)
            rationale = f"Only {good_pct:.1f}% pass; too many valid masks rejected. Relaxing to median - 0.05."
        # Optimal range
        else:
            recommended = current_threshold
            rationale = f"Current threshold is well-balanced ({good_pct:.1f}% pass rate)."
        
        recommended = max(0.20, min(recommended, 0.70))  # Sensible bounds
        
        return ThresholdRecommendation(
            parameter="min_coverage_ratio",
            current_value=current_threshold,
            recommended_value=round(recommended, 3),
            rationale=rationale,
            impact="high"
        )
    
    @staticmethod
    def recommend_sam_iou_threshold(report: dict) -> Optional[ThresholdRecommendation]:
        """Recommend SAM IoU threshold based on distribution."""
        sam_stats = report.get("sam_analysis", {})
        current_threshold = sam_stats.get("threshold", 0.60)
        good_pct = sam_stats.get("good_sam_percentage", 0)
        mean_iou = sam_stats.get("mean", None)
        median_iou = sam_stats.get("median", None)
        
        if mean_iou is None or median_iou is None:
            return None  # No SAM scores available
        
        # If >80% pass, threshold is too permissive
        if good_pct > 80:
            recommended = min(0.75, median_iou - 0.05)
            rationale = f"{good_pct:.1f}% pass current threshold. Recommending stricter {recommended:.3f}."
        # If <30% pass, threshold is too strict
        elif good_pct < 30:
            recommended = max(0.40, median_iou - 0.10)
            rationale = f"Only {good_pct:.1f}% pass; too strict. Relaxing to {recommended:.3f}."
        # Optimal
        else:
            recommended = current_threshold
            rationale = f"Current threshold is well-calibrated ({good_pct:.1f}% pass rate)."
        
        recommended = max(0.30, min(recommended, 0.90))
        
        return ThresholdRecommendation(
            parameter="sam_iou_threshold",
            current_value=current_threshold,
            recommended_value=round(recommended, 3),
            rationale=rationale,
            impact="medium"
        )
    
    @staticmethod
    def recommend_confirmation_blur_threshold(report: dict) -> ThresholdRecommendation:
        """Recommend stricter confirmation preview blur threshold."""
        blur_stats = report.get("blur_analysis", {})
        p75 = blur_stats.get("distribution_percentiles", {}).get("p75", 100)
        p90 = blur_stats.get("distribution_percentiles", {}).get("p90", 120)
        
        # Use p75 for operator confirmation
        recommended = p75
        
        return ThresholdRecommendation(
            parameter="confirmation.blur_threshold",
            current_value=100.0,
            recommended_value=round(recommended, 1),
            rationale=f"Use p75 ({recommended:.1f}) for stricter operator review (rejects bottom 25%).",
            impact="medium"
        )
    
    @staticmethod
    def generate_recommendations(report: dict) -> dict:
        """Generate all recommendations."""
        recommendations = []
        
        # Core recommendations
        blur_rec = ConfigOptimizer.recommend_blur_threshold(report)
        recommendations.append(blur_rec)
        
        cov_rec = ConfigOptimizer.recommend_coverage_threshold(report)
        recommendations.append(cov_rec)
        
        sam_rec = ConfigOptimizer.recommend_sam_iou_threshold(report)
        if sam_rec:
            recommendations.append(sam_rec)
        
        conf_blur_rec = ConfigOptimizer.recommend_confirmation_blur_threshold(report)
        recommendations.append(conf_blur_rec)
        
        # Convert to dict
        recs_dict = {
            "recommendations": [
                {
                    "parameter": r.parameter,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "rationale": r.rationale,
                    "impact": r.impact,
                }
                for r in recommendations
            ]
        }
        
        return recs_dict


# Example usage / testing
if __name__ == "__main__":
    # Sample report for demonstration
    sample_report = {
        "total_images": 150,
        "blur_analysis": {
            "threshold": 40.0,
            "sharp_count": 105,
            "sharp_percentage": 70.0,
            "mean": 52.34,
            "median": 48.92,
            "min": 15.2,
            "max": 125.6,
            "distribution_percentiles": {
                "p10": 28.5,
                "p25": 38.2,
                "p50": 48.92,
                "p75": 62.1,
                "p90": 78.5,
            }
        },
        "coverage_analysis": {
            "threshold": 0.35,
            "good_coverage_count": 132,
            "good_coverage_percentage": 88.0,
            "mean": 0.58,
            "median": 0.62,
            "min": 0.12,
            "max": 0.98,
        },
        "sam_analysis": {
            "threshold": 0.60,
            "images_with_sam_scores": 140,
            "good_sam_count": 98,
            "good_sam_percentage": 70.0,
            "mean": 0.652,
            "median": 0.68,
            "min": 0.35,
            "max": 0.95,
        },
    }
    
    recommendations = ConfigOptimizer.generate_recommendations(sample_report)
    print(json.dumps(recommendations, indent=2))

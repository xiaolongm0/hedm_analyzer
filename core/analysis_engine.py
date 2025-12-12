"""
Analysis engine for HEDM X-ray image statistical analysis
Calculates frame statistics, ROI statistics, saturation analysis, and histograms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ROI:
    """Region of Interest definition"""
    name: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        return (self.x_start, self.y_start, self.x_end, self.y_end)
    
    @property
    def width(self) -> int:
        return self.x_end - self.x_start
    
    @property
    def height(self) -> int:
        return self.y_end - self.y_start
    
    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass
class Statistics:
    """Statistical results container"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    sum: float
    count: int
    percentiles: Dict[str, float]
    
    def to_dict(self) -> dict:
        return {
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'sum': self.sum,
            'count': self.count,
            'percentiles': self.percentiles
        }

class AnalysisEngine:
    """Core analysis engine for HEDM data"""
    
    def __init__(self):
        self.rois = []
        self.threshold = 0
        self.mask = None
        self.saturation_threshold = 65535  # Default for 16-bit detectors
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_roi(self, roi: ROI) -> None:
        """Add a region of interest"""
        self.rois.append(roi)
        self.logger.info(f"Added ROI '{roi.name}': {roi.coordinates}")
    
    def remove_roi(self, name: str) -> bool:
        """Remove ROI by name"""
        for i, roi in enumerate(self.rois):
            if roi.name == name:
                del self.rois[i]
                self.logger.info(f"Removed ROI '{name}'")
                return True
        return False
    
    def clear_rois(self) -> None:
        """Clear all ROIs"""
        self.rois.clear()
        self.logger.info("Cleared all ROIs")
    
    def set_threshold(self, threshold: float) -> None:
        """Set lower intensity threshold for analysis

        Pixels with intensity < threshold will be excluded from:
        - Statistics calculation (mean, median, std, etc.)
        - Histogram generation
        """
        self.threshold = threshold
    
    def set_mask(self, mask: np.ndarray) -> None:
        """Set pixel mask for excluding regions"""
        self.mask = mask
    
    def set_saturation_threshold(self, threshold: int) -> None:
        """Set saturation threshold (typically 2^n - 1 for n-bit detector)"""
        self.saturation_threshold = threshold
    
    def calculate_statistics(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Statistics:
        """Calculate basic statistics for data array

        Applies lower threshold to exclude pixels below the threshold value.
        Only pixels >= threshold are included in statistics calculation.
        """
        if mask is not None:
            # Apply mask (True = valid pixels)
            valid_data = data[mask]
        else:
            valid_data = data.flatten()

        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]

        # Apply lower threshold to exclude low-intensity pixels
        if self.threshold > 0:
            valid_data = valid_data[valid_data >= self.threshold]

        if len(valid_data) == 0:
            return Statistics(0, 0, 0, 0, 0, 0, 0, {})

        return Statistics(
            mean=float(np.mean(valid_data)),
            median=float(np.median(valid_data)),
            std=float(np.std(valid_data)),
            min=float(np.min(valid_data)),
            max=float(np.max(valid_data)),
            sum=float(np.sum(valid_data)),
            count=len(valid_data),
            percentiles={}  # No percentiles for speed
        )
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Statistics]:
        """Analyze a single frame for all ROIs plus overall statistics"""
        results = {}
        
        # Overall frame statistics
        frame_mask = self.mask if self.mask is not None else None
        results['overall'] = self.calculate_statistics(frame, frame_mask)
        
        # ROI statistics
        for roi in self.rois:
            roi_data = frame[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            
            # Apply mask to ROI if available
            roi_mask = None
            if self.mask is not None:
                roi_mask = self.mask[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            
            results[roi.name] = self.calculate_statistics(roi_data, roi_mask)
        
        return results
    
    
    def calculate_saturation_analysis(self, frame: np.ndarray) -> Dict[str, Union[int, float]]:
        """Analyze pixel saturation for a single frame"""
        saturated_pixels = frame >= self.saturation_threshold

        total_pixels = frame.size
        saturated_count = np.sum(saturated_pixels)
        saturation_percentage = (saturated_count / total_pixels) * 100 if total_pixels > 0 else 0

        return {
            'saturated_pixels': int(saturated_count),
            'total_pixels': int(total_pixels),
            'saturation_percentage': float(saturation_percentage),
            'saturation_threshold': self.saturation_threshold
        }
    
    def calculate_histogram(self, frame: np.ndarray, bins: Union[int, str] = 'auto',
                          roi: Optional[ROI] = None) -> np.ndarray:
        """Calculate histogram data for a single frame - returns flattened data for matplotlib

        Returns the raw data array (not histogram bins) so matplotlib can calculate
        the PDF properly with density=True.

        Applies lower threshold to exclude pixels below the threshold value.
        Only pixels >= threshold are included in histogram.
        """
        # Extract ROI if specified
        if roi is not None:
            hist_data = frame[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            # Apply mask to ROI if available
            if self.mask is not None:
                roi_mask = self.mask[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
                hist_data = hist_data[roi_mask]
        else:
            # Full frame
            hist_data = frame
            # Apply mask if available
            if self.mask is not None:
                hist_data = hist_data[self.mask]

        # Flatten and remove NaN values
        hist_data = hist_data.flatten()
        hist_data = hist_data[~np.isnan(hist_data)]

        # Apply threshold if set
        if self.threshold > 0:
            hist_data = hist_data[hist_data >= self.threshold]

        return hist_data

    def analyze_all_frames_stats(self, data: np.ndarray, skip_frames: int = 0) -> Dict:
        """Analyze all frames and return per-frame statistics (no histograms)

        This is optimized for saving statistics to file without storing
        large histogram data in memory.

        Args:
            data: Input data array (frames x height x width)
            skip_frames: Number of initial frames to skip (default: 0)
        """
        total_frames = data.shape[0]
        start_frame = skip_frames
        frames_to_analyze = total_frames - skip_frames

        if skip_frames > 0:
            self.logger.info(f"Skipping first {skip_frames} frame(s)")
            self.logger.info(f"Analyzing {frames_to_analyze} frames (frame {start_frame + 1} to {total_frames})...")
        else:
            self.logger.info(f"Analyzing {total_frames} frames...")

        results = {
            'analysis_parameters': {
                'threshold': self.threshold,
                'saturation_threshold': self.saturation_threshold,
                'num_rois': len(self.rois),
                'roi_names': [roi.name for roi in self.rois],
                'mask_applied': self.mask is not None,
                'skip_frames': skip_frames
            },
            'data_info': {
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'total_frames': int(total_frames),
                'analyzed_frames': int(frames_to_analyze),
                'start_frame': int(start_frame),
                'frame_width': int(data.shape[2]),
                'frame_height': int(data.shape[1])
            },
            'per_frame_statistics': []
        }

        # Analyze each frame (starting from skip_frames)
        for frame_idx in range(start_frame, total_frames):
            frame = data[frame_idx]

            # Calculate frame statistics
            frame_stats = self.analyze_frame(frame)
            saturation_stats = self.calculate_saturation_analysis(frame)

            frame_result = {
                'frame_index': int(frame_idx),
                'statistics': {name: stats.to_dict() for name, stats in frame_stats.items()},
                'saturation': saturation_stats
            }

            results['per_frame_statistics'].append(frame_result)

            # Log progress every 10%
            frames_processed = frame_idx - start_frame + 1
            if frames_processed % max(1, frames_to_analyze // 10) == 0:
                progress = (frames_processed / frames_to_analyze) * 100
                self.logger.info(f"Progress: {progress:.0f}% ({frames_processed}/{frames_to_analyze} frames)")

        self.logger.info("All frames analysis complete!")
        return results


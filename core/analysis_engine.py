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
        """Set intensity threshold for analysis"""
        self.threshold = threshold
    
    def set_mask(self, mask: np.ndarray) -> None:
        """Set pixel mask for excluding regions"""
        self.mask = mask
    
    def set_saturation_threshold(self, threshold: int) -> None:
        """Set saturation threshold (typically 2^n - 1 for n-bit detector)"""
        self.saturation_threshold = threshold
    
    def calculate_statistics(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Statistics:
        """Calculate comprehensive statistics for data array"""
        if mask is not None:
            # Apply mask (True = valid pixels)
            valid_data = data[mask]
        else:
            valid_data = data.flatten()
        
        # Remove NaN values
        valid_data = valid_data[~np.isnan(valid_data)]
        
        if len(valid_data) == 0:
            return Statistics(0, 0, 0, 0, 0, 0, 0, {})
        
        # Calculate percentiles
        percentiles = {
            '1%': np.percentile(valid_data, 1),
            '5%': np.percentile(valid_data, 5),
            '10%': np.percentile(valid_data, 10),
            '25%': np.percentile(valid_data, 25),
            '75%': np.percentile(valid_data, 75),
            '90%': np.percentile(valid_data, 90),
            '95%': np.percentile(valid_data, 95),
            '99%': np.percentile(valid_data, 99)
        }
        
        return Statistics(
            mean=float(np.mean(valid_data)),
            median=float(np.median(valid_data)),
            std=float(np.std(valid_data)),
            min=float(np.min(valid_data)),
            max=float(np.max(valid_data)),
            sum=float(np.sum(valid_data)),
            count=len(valid_data),
            percentiles=percentiles
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
    
    def analyze_all_frames(self, data: np.ndarray) -> Dict[str, Dict[str, Union[List, Statistics]]]:
        """Analyze all frames in dataset"""
        results = {
            'frame_statistics': defaultdict(list),  # Per-frame stats for each ROI
            'overall_statistics': {},  # Aggregated stats across all frames
            'frame_indices': list(range(data.shape[0]))
        }
        
        # Analyze each frame
        for frame_idx in range(data.shape[0]):
            frame_stats = self.analyze_frame(data[frame_idx])
            
            # Store per-frame results
            for roi_name, stats in frame_stats.items():
                results['frame_statistics'][roi_name].append(stats)
        
        # Calculate overall statistics across all frames
        for roi_name in results['frame_statistics'].keys():
            if roi_name == 'overall':
                # For overall, use all data
                overall_data = data
                overall_mask = self.mask
            else:
                # For ROI, extract all ROI data
                roi = next(r for r in self.rois if r.name == roi_name)
                overall_data = data[:, roi.y_start:roi.y_end, roi.x_start:roi.x_end]
                overall_mask = None
                if self.mask is not None:
                    overall_mask = self.mask[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            
            results['overall_statistics'][roi_name] = self.calculate_statistics(overall_data, overall_mask)
        
        return results
    
    def calculate_saturation_analysis(self, data: np.ndarray) -> Dict[str, Union[int, float]]:
        """Analyze pixel saturation across dataset"""
        saturated_pixels = data >= self.saturation_threshold
        
        total_pixels = data.size
        saturated_count = np.sum(saturated_pixels)
        saturation_percentage = (saturated_count / total_pixels) * 100
        
        # Per-frame saturation
        frames_with_saturation = 0
        max_frame_saturation = 0
        frame_saturation_percentages = []
        
        for frame_idx in range(data.shape[0]):
            frame_saturated = np.sum(saturated_pixels[frame_idx])
            frame_pixels = data[frame_idx].size
            frame_sat_pct = (frame_saturated / frame_pixels) * 100
            frame_saturation_percentages.append(frame_sat_pct)
            
            if frame_saturated > 0:
                frames_with_saturation += 1
            
            max_frame_saturation = max(max_frame_saturation, frame_sat_pct)
        
        return {
            'total_saturated_pixels': int(saturated_count),
            'total_pixels': int(total_pixels),
            'saturation_percentage': float(saturation_percentage),
            'frames_with_saturation': int(frames_with_saturation),
            'total_frames': int(data.shape[0]),
            'max_frame_saturation_pct': float(max_frame_saturation),
            'frame_saturation_percentages': frame_saturation_percentages,
            'saturation_threshold': self.saturation_threshold
        }
    
    def calculate_histogram(self, data: np.ndarray, bins: Union[int, str] = 'auto', 
                          roi: Optional[ROI] = None, frame_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate histogram of intensity values"""
        if frame_idx is not None:
            # Single frame analysis
            if roi is not None:
                hist_data = data[frame_idx, roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            else:
                hist_data = data[frame_idx]
        else:
            # All frames analysis
            if roi is not None:
                hist_data = data[:, roi.y_start:roi.y_end, roi.x_start:roi.x_end]
            else:
                hist_data = data
        
        # Apply mask if available
        if self.mask is not None:
            if roi is not None:
                roi_mask = self.mask[roi.y_start:roi.y_end, roi.x_start:roi.x_end]
                if frame_idx is not None:
                    hist_data = hist_data[roi_mask]
                else:
                    # Broadcast mask across frames
                    mask_3d = np.broadcast_to(roi_mask, hist_data.shape)
                    hist_data = hist_data[mask_3d]
            else:
                if frame_idx is not None:
                    hist_data = hist_data[self.mask]
                else:
                    mask_3d = np.broadcast_to(self.mask, hist_data.shape)
                    hist_data = hist_data[mask_3d]
        
        # Flatten and remove NaN values
        hist_data = hist_data.flatten()
        hist_data = hist_data[~np.isnan(hist_data)]
        
        # Apply threshold if set
        if self.threshold > 0:
            hist_data = hist_data[hist_data >= self.threshold]
        
        if len(hist_data) == 0:
            return np.array([]), np.array([])
        
        # Calculate histogram
        counts, bin_edges = np.histogram(hist_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return counts, bin_centers
    
    def generate_report(self, data: np.ndarray) -> Dict:
        """Generate comprehensive analysis report"""
        self.logger.info("Generating comprehensive analysis report...")
        
        # Analyze all frames
        frame_analysis = self.analyze_all_frames(data)
        
        # Saturation analysis
        saturation_analysis = self.calculate_saturation_analysis(data)
        
        # Generate histograms for overall and each ROI
        histograms = {}
        
        # Overall histogram
        counts, bins = self.calculate_histogram(data)
        histograms['overall'] = {'counts': counts.tolist(), 'bins': bins.tolist()}
        
        # ROI histograms
        for roi in self.rois:
            counts, bins = self.calculate_histogram(data, roi=roi)
            histograms[roi.name] = {'counts': counts.tolist(), 'bins': bins.tolist()}
        
        report = {
            'analysis_parameters': {
                'threshold': self.threshold,
                'saturation_threshold': self.saturation_threshold,
                'num_rois': len(self.rois),
                'roi_names': [roi.name for roi in self.rois],
                'mask_applied': self.mask is not None
            },
            'data_info': {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'total_pixels': int(data.size)
            },
            'frame_statistics': {
                name: [stats.to_dict() for stats in stats_list] 
                for name, stats_list in frame_analysis['frame_statistics'].items()
            },
            'overall_statistics': {
                name: stats.to_dict() 
                for name, stats in frame_analysis['overall_statistics'].items()
            },
            'saturation_analysis': saturation_analysis,
            'histograms': histograms
        }
        
        self.logger.info("Analysis report generated successfully")
        return report
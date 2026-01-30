"""
Parameter Optimizer for HEDM Analyzer
Analyzes experimental calibration data to recommend optimal exposure parameters
using Beer-Lambert law regression
"""

import os
import re
import numpy as np
import tifffile
import logging
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Optional, Dict, Any


class ParameterOptimizer:
    """Analyzes experimental data to recommend optimal exposure parameters"""

    # Constants from experiment_parameter_recommendation.py
    AVAILABLE_THICKNESSES_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.39, 4.78, 7.14, 9.53, 11.91, 14.30, 16.66]
    SATURATION_LIMIT = 65535
    SAFETY_MARGIN = 0.90
    FRAMES_TO_CHECK = 10

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the parameter optimizer"""
        self.logger = logger or logging.getLogger(__name__)

    def parse_folder_info(self, folder_name: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Extract attenuation index and exposure time from folder name.

        Expected format: sample_atten-{idx}_t-{time}
        Example: sample_atten-0_t-0.1

        Args:
            folder_name: Name of the folder to parse

        Returns:
            Tuple of (atten_idx, exposure_time) or (None, None) if not valid
        """
        match = re.search(r"atten-(\d+)_t-([\d\.]+)", folder_name)
        if match:
            return int(match.group(1)), float(match.group(2))
        return None, None

    def get_max_intensity(self, folder_path: Path) -> float:
        """
        Find maximum pixel value in a subset of TIFF images in the folder.

        Samples every Nth frame to limit memory usage.

        Args:
            folder_path: Path to the folder containing TIFF files

        Returns:
            Global maximum intensity value across sampled frames
        """
        tif_files = sorted(list(folder_path.glob("*.tif")))
        if not tif_files:
            return 0

        step = max(1, len(tif_files) // self.FRAMES_TO_CHECK)
        subset = tif_files[::step]

        global_max = 0
        for f in subset:
            try:
                img = tifffile.imread(f)
                global_max = max(global_max, np.max(img))
            except Exception as e:
                self.logger.debug(f"Could not read {f}: {str(e)}")

        return global_max

    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Analyze experimental calibration data directory and fit Beer-Lambert law.

        Workflow:
        1. Scan directory for folders matching pattern: sample_atten-{idx}_t-{time}
        2. Extract max intensity from TIFF files in each folder
        3. Calculate intensity rate = max_intensity / exposure_time
        4. Filter saturated data (>= SATURATION_LIMIT - 100)
        5. Group rates by thickness
        6. Perform linear regression on log-transformed data:
           ln(rate) = -μ × thickness + ln(I₀)
        7. Generate predictions for all available thicknesses

        Args:
            directory_path: Path to directory containing calibration folders

        Returns:
            Dictionary with structure:
            {
                'success': bool,
                'error': Optional[str],
                'data_points': {thickness: [rates]},
                'fit_results': {'mu': float, 'I0': float, 'r_squared': float},
                'predictions': [
                    {'atten_idx': int, 'thickness': float, 'rate': float, 'max_time': float},
                    ...
                ]
            }
        """
        root_path = Path(directory_path)

        # Check directory exists
        if not root_path.is_dir():
            return {
                'success': False,
                'error': f'Directory not found: {directory_path}',
                'data_points': {},
                'fit_results': {},
                'predictions': []
            }

        # 1. HARVEST DATA
        data_points = defaultdict(list)

        self.logger.info(f"Scanning directory: {directory_path}")

        folders = [f for f in root_path.iterdir() if f.is_dir()]

        for folder in folders:
            atten_idx, exposure_time = self.parse_folder_info(folder.name)

            # Validation
            if atten_idx is None or exposure_time is None:
                continue
            if atten_idx >= len(self.AVAILABLE_THICKNESSES_MM):
                self.logger.debug(f"Skipping {folder.name}: atten_idx {atten_idx} out of range")
                continue

            thickness = self.AVAILABLE_THICKNESSES_MM[atten_idx]
            max_val = self.get_max_intensity(folder)

            # Filter Saturated Data
            if max_val >= (self.SATURATION_LIMIT - 100):
                self.logger.info(f"Skipping {folder.name}: Saturated ({max_val})")
                continue

            if exposure_time > 0 and max_val > 0:
                rate = max_val / exposure_time
                data_points[thickness].append(rate)
                self.logger.info(f"Read {folder.name}: Thick={thickness}mm, Rate={rate:.1f} cts/s")

        # Check if we have any data
        if not data_points:
            return {
                'success': False,
                'error': 'No valid calibration folders found',
                'data_points': {},
                'fit_results': {},
                'predictions': []
            }

        # 2. PREPARE FOR FITTING
        valid_thicknesses = []
        valid_log_rates = []

        self.logger.info("Processing data points for regression")

        for thick, rates in sorted(data_points.items()):
            avg_rate = np.mean(rates)
            log_rate = np.log(avg_rate)

            valid_thicknesses.append(thick)
            valid_log_rates.append(log_rate)
            self.logger.debug(f"Thickness: {thick}mm | Avg Rate: {avg_rate:.1f} | Ln(Rate): {log_rate:.4f}")

        # Check if we have enough data points
        if len(valid_thicknesses) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 data points for fitting',
                'data_points': dict(data_points),
                'fit_results': {},
                'predictions': []
            }

        # 3. PERFORM LINEAR REGRESSION
        # Fit: y = mx + c where y = ln(Rate), x = Thickness
        # Slope (m) = -mu, Intercept (c) = ln(I0)

        valid_thicknesses = np.array(valid_thicknesses)
        valid_log_rates = np.array(valid_log_rates)

        coeffs = np.polyfit(valid_thicknesses, valid_log_rates, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        calculated_mu = -slope
        calculated_I0 = np.exp(intercept)

        # Calculate R-squared for goodness of fit
        y_pred = slope * valid_thicknesses + intercept
        ss_res = np.sum((valid_log_rates - y_pred) ** 2)
        ss_tot = np.sum((valid_log_rates - np.mean(valid_log_rates)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.logger.info(f"Fitted parameters: μ={calculated_mu:.4f}/mm, I₀={calculated_I0:.2e} cts/s, R²={r_squared:.4f}")

        if calculated_mu < 0:
            self.logger.warning("Calculated mu is negative. Data might be noisy or inconsistent.")

        # 4. EXTRAPOLATE & PREDICT
        target_counts = self.SATURATION_LIMIT * self.SAFETY_MARGIN
        predictions = []

        for i, thick in enumerate(self.AVAILABLE_THICKNESSES_MM):
            predicted_rate = calculated_I0 * np.exp(-calculated_mu * thick)
            max_time = target_counts / predicted_rate if predicted_rate > 0 else float('inf')

            predictions.append({
                'atten_idx': i,
                'thickness': thick,
                'rate': float(predicted_rate),
                'max_time': float(max_time)
            })

        return {
            'success': True,
            'error': None,
            'data_points': dict(data_points),
            'fit_results': {
                'mu': float(calculated_mu),
                'I0': float(calculated_I0),
                'r_squared': float(r_squared)
            },
            'predictions': predictions
        }

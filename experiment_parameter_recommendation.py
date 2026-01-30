import os
import re
import numpy as np
import tifffile
from pathlib import Path
from collections import defaultdict
 
# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_DIRECTORY = "./experiment_data"
 
# The COMPLETE list of Cu thicknesses available on your wheel/slider (in mm).
# The index in this list corresponds to the 'atten-X' number in the folder name.
# Example: atten-0 -> 0.0mm, atten-1 -> 0.1mm, etc.
AVAILABLE_THICKNESSES_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.39, 4.78, 7.14, 9.53, 11.91, 14.30, 16.66]
 
SATURATION_LIMIT = 65535
SAFETY_MARGIN = 0.90  # Target 90% of saturation
FRAMES_TO_CHECK = 10
 
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
 
def parse_folder_info(folder_name):
    """Extracts atten index and time from folder string."""
    match = re.search(r"atten-(\d+)_t-([\d\.]+)", folder_name)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None, None
 
def get_max_intensity(folder_path):
    """Finds max pixel value in a subset of images in the folder."""
    tif_files = sorted(list(folder_path.glob("*.tif")))
    if not tif_files: return 0
    
    step = max(1, len(tif_files) // FRAMES_TO_CHECK)
    subset = tif_files[::step]
    
    global_max = 0
    for f in subset:
        try:
            img = tifffile.imread(f)
            global_max = max(global_max, np.max(img))
        except: pass
    return global_max
 
# =============================================================================
# MAIN LOGIC
# =============================================================================
 
def main():
    root_path = Path(MASTER_DIRECTORY)
    
    # 1. HARVEST DATA
    # We need pairs of (Thickness, IntensityRate)
    # We store multiple rates if available to average them later
    data_points = defaultdict(list)
 
    print(f"--- Scanning Data in {MASTER_DIRECTORY} ---")
    
    folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    for folder in folders:
        atten_idx, exposure_time = parse_folder_info(folder.name)
        
        # Validation
        if atten_idx is None or exposure_time is None: continue
        if atten_idx >= len(AVAILABLE_THICKNESSES_MM): continue
        
        thickness = AVAILABLE_THICKNESSES_MM[atten_idx]
        max_val = get_max_intensity(folder)
        
        # Filter Saturated Data
        # We CANNOT use saturated data for fitting because the rate is clipped.
        # It distorts the curve. We strictly ignore it for the physics calculation.
        if max_val >= (SATURATION_LIMIT - 100):
            print(f"Skipping {folder.name}: Saturated ({max_val})")
            continue
            
        if exposure_time > 0 and max_val > 0:
            rate = max_val / exposure_time
            data_points[thickness].append(rate)
            print(f"Read {folder.name}: Thick={thickness}mm, Rate={rate:.1f} cts/s")
 
    # 2. PREPARE FOR FITTING
    # Average the rates for each thickness to get one clean point per thickness
    valid_thicknesses = []
    valid_log_rates = []
    
    print("\n--- Processing Data Points ---")
    for thick, rates in data_points.items():
        avg_rate = np.mean(rates)
        # We take the Natural Log (ln) of the rate for linear fitting
        log_rate = np.log(avg_rate)
        
        valid_thicknesses.append(thick)
        valid_log_rates.append(log_rate)
        print(f"Thickness: {thick}mm | Avg Rate: {avg_rate:.1f} | Ln(Rate): {log_rate:.4f}")
 
    if len(valid_thicknesses) < 2:
        print("\nERROR: Not enough data points to fit a curve.")
        print("Need at least 2 different thicknesses with non-saturated data.")
        return
 
    # 3. PERFORM LINEAR REGRESSION (The "Learning" Step)
    # We fit: y = mx + c
    # y = ln(Rate), x = Thickness
    # Slope (m) = -mu
    # Intercept (c) = ln(I0)
    
    slope, intercept = np.polyfit(valid_thicknesses, valid_log_rates, 1)
    
    calculated_mu = -slope
    calculated_I0 = np.exp(intercept)
    
    print(f"\n--- FITTED PHYSICS MODEL ---")
    print(f"Calculated Attenuation Coeff (mu): {calculated_mu:.4f} /mm")
    print(f"Calculated Unattenuated Flux (I0): {calculated_I0:.2e} cts/s")
    
    # Check goodness of fit (optional but helpful)
    # If mu is negative, something is wrong (intensity increased with thickness)
    if calculated_mu < 0:
        print("WARNING: Calculated mu is negative. Data might be noisy or inconsistent.")
 
    # 4. EXTRAPOLATE & PREDICT
    print(f"\n--- PREDICTED MAX EXPOSURE TIMES ---")
    print(f"{'Atten Index':<12} | {'Thickness':<10} | {'Pred. Rate (cts/s)':<20} | {'MAX TIME (s)':<15}")
    print("-" * 65)
    
    target_counts = SATURATION_LIMIT * SAFETY_MARGIN
    
    for i, thick in enumerate(AVAILABLE_THICKNESSES_MM):
        # Apply the learned law: Rate = I0 * exp(-mu * x)
        predicted_rate = calculated_I0 * np.exp(-calculated_mu * thick)
        
        # Calculate Max Time
        max_time = target_counts / predicted_rate
        
        print(f"atten-{i:<6} | {thick:<10} | {predicted_rate:<20.2e} | {max_time:.4f}")
 
if __name__ == "__main__":
    main()
 
import os
import numpy as np
import tifffile
from pathlib import Path
import shutil
 
# =============================================================================
# CONFIGURATION FOR SYNTHETIC DATA
# =============================================================================
 
OUTPUT_DIR = "./experiment_data"
 
# Image Properties
IMG_SHAPE = (2048, 2048)
BIT_DEPTH = 65535  # 16-bit unsigned integer
 
# Scan Parameters
START_ANGLE = 0
END_ANGLE = 360
STEP_SIZE = 0.5
TOTAL_PROJECTIONS = int((END_ANGLE - START_ANGLE) / STEP_SIZE)
 
# OPTIMIZATION: To save disk space and time for this test,
# we will only save every Nth projection.
# Set to 1 to generate the full 720 images (Warning: ~6GB per scan).
SAVE_EVERY_NTH_FRAME = 20
 
# =============================================================================
# HIDDEN PHYSICS CONSTANTS (The "Ground Truth")
# =============================================================================
 
# We assume the source produces this many counts per second in the direct beam
# (This is the I0 your analysis script should try to discover)
TRUE_SOURCE_FLUX = 50000.0  # counts per second
 
# The "True" attenuation coefficient of the Copper filter for this X-ray energy
# (This is the mu your analysis script should try to discover)
TRUE_MU_COPPER = 0.85  # 1/mm
 
# Available Copper thicknesses on the wheel (must match your analysis script)
THICKNESS_MAP = {
    0: 0.0,
    1: 0.1,
    2: 0.2,
    3: 0.5,
    4: 1.0,
    5: 2.0
}
 
# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================
 
def create_phantom(shape):
    """
    Creates a simple 2D phantom: A cylinder in the center.
    Returns a mask where 1.0 = air, < 1.0 = sample attenuation.
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    radius = shape[0] // 4
    
    # Create a circle mask
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Base transmission is 1.0 (Air)
    transmission_image = np.ones(shape, dtype=np.float32)
    
    # Inside the circle, the sample absorbs 50% of X-rays
    transmission_image[mask] = 0.5
    
    return transmission_image
 
def generate_scan(atten_index, exposure_time, phantom_base):
    """
    Generates a folder of TIFFs for a specific attenuation and time.
    """
    copper_thickness = THICKNESS_MAP.get(atten_index, 0.0)
    
    # 1. Calculate Beam Intensity after Copper Filter (Beer-Lambert Law)
    # I_beam = I0 * exp(-mu * thickness)
    beam_intensity_rate = TRUE_SOURCE_FLUX * np.exp(-TRUE_MU_COPPER * copper_thickness)
    
    # 2. Calculate Expected Counts for this exposure time
    # This is the value in the "Air" region of the image
    expected_air_counts = beam_intensity_rate * exposure_time
    
    folder_name = f"sample_atten-{atten_index}_t-{exposure_time}"
    folder_path = Path(OUTPUT_DIR) / folder_name
    
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True)
    
    print(f"Generating: {folder_name:<30} | Copper: {copper_thickness}mm | Exp. Air Counts: {expected_air_counts:.0f}")
 
    # Generate frames
    for i in range(0, TOTAL_PROJECTIONS, SAVE_EVERY_NTH_FRAME):
        # Apply the phantom attenuation to the beam
        # (Air regions stay at expected_air_counts, Sample regions get darker)
        ideal_image = phantom_base * expected_air_counts
        
        # Add Poisson Noise (Photon counting statistics)
        # Poisson noise is sqrt(N).
        noisy_image = np.random.poisson(ideal_image).astype(np.float32)
        
        # Simulate Detector Saturation
        # Clip values to the 16-bit limit
        noisy_image = np.clip(noisy_image, 0, BIT_DEPTH)
        
        # Convert to integer for saving
        final_image = noisy_image.astype(np.uint16)
        
        # Save file
        filename = f"proj_{i:04d}.tif"
        tifffile.imwrite(folder_path / filename, final_image)
 
# =============================================================================
# MAIN EXECUTION
# =============================================================================
 
def main():
    print("--- Starting Synthetic Data Generation ---")
    print(f"Image Size: {IMG_SHAPE}")
    print(f"Simulating Source Flux: {TRUE_SOURCE_FLUX} cts/s")
    print(f"Simulating Copper Mu:   {TRUE_MU_COPPER} /mm")
    print("-" * 60)
 
    # Pre-calculate the phantom shape to speed up the loop
    phantom_base = create_phantom(IMG_SHAPE)
 
    # Define the scenarios to generate
    # We create a mix of saturated and good data to test your logic.
    
    scenarios = [
        # --- Attenuation 0 (No filter) ---
        # 2.0s should be VERY saturated (50k * 2 = 100k > 65k)
        (0, 2.0),
        # 0.5s should be good (50k * 0.5 = 25k)
        (0, 0.5),
        
        # --- Attenuation 1 (0.1mm) ---
        # Flux drops to ~46k. 1.5s -> ~69k (Saturated)
        (1, 1.5),
        # 1.0s -> ~46k (Good)
        (1, 1.0),
        
        # --- Attenuation 3 (0.5mm) ---
        # Flux drops to ~32k.
        (3, 1.0),
        
        # --- Attenuation 5 (2.0mm) ---
        # Flux drops massively to ~9k.
        (5, 2.0)
    ]
 
    for atten_idx, exp_time in scenarios:
        generate_scan(atten_idx, exp_time, phantom_base)
 
    print("-" * 60)
    print("Generation Complete.")
    print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("You can now run your analysis script pointing to this folder.")
 
if __name__ == "__main__":
    main()
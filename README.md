# HEDM X-ray Image Analyzer

A comprehensive Python application for analyzing High Energy Diffraction Microscopy (HEDM) X-ray detector data with statistical analysis, ROI management, and scan parameter optimization.

## Features

### Core Functionality
- **Multi-format Data Loading**: Support for HDF5 files, image sequences (PNG/TIFF), and single images
- **Statistical Analysis**: Comprehensive per-frame and per-ROI statistics (mean, median, std, percentiles)
- **ROI Management**: Interactive region-of-interest selection and analysis
- **Saturation Detection**: Automatic detection and reporting of saturated pixels
- **Histogram Generation**: Intensity distribution analysis with customizable binning
- **X-ray Physics**: Attenuation calculations and scan parameter recommendations

### Analysis Outputs
- **Statistics**: Per-frame and overall statistics for each ROI and global data
- **Histograms**: Intensity distribution plots for overall data and individual ROIs
- **Saturation Analysis**: Count and percentage of saturated pixels
- **Attenuation Reports**: Current attenuation factors based on material filters and X-ray energy
- **Parameter Recommendations**: Optimized exposure times and attenuation settings

## Installation

### Requirements
```bash
pip install numpy scipy pillow h5py matplotlib scikit-image
```

See `requirements.txt` for detailed version requirements.

### Setup
```bash
cd hedm_analyzer
python3 main.py
```

## Usage

### GUI Application
The main interface provides:
1. **Data Input Panel**: Load HDF5 files or image sequences
2. **Parameter Controls**: Set threshold, saturation limits, and scan conditions
3. **ROI Management**: Interactive region selection on image viewer
4. **Analysis Controls**: Run analysis and export results
5. **Visualization**: Real-time histograms and statistical plots

### Basic Workflow
1. Load data (HDF5 file or image sequence)
2. Set analysis parameters (threshold, saturation limit)
3. Define ROIs by clicking and dragging on images
4. Configure scan conditions (X-ray energy, exposure time)
5. Run analysis to generate comprehensive report
6. Export results to JSON/CSV formats

### Input Parameters
- **File Input**: HDF5 files (`*.h5`) or image directories
- **Threshold**: Minimum intensity for analysis inclusion
- **Mask File**: Optional pixel exclusion mask (PNG/TIFF/text)
- **Scan Conditions**:
  - X-ray energy (keV)
  - Exposure time (seconds) 
  - Attenuation filter settings
  - Beam current and detector distance

## Data Format Support

### HDF5 Files (Recommended)
Expected structure following NeXus/TomoPy conventions:
```
/exchange/data          # Projection images (frames, height, width)
/exchange/bright        # Bright field images
/exchange/dark          # Dark field images
/measurement/           # Experimental metadata
```

### Image Sequences
- Supported formats: PNG, TIFF
- Files should be numbered sequentially
- All images must have the same dimensions

## Core Components

### DataHandler (`core/data_handler.py`)
- Multi-format file loading
- Metadata extraction from HDF5
- Mask application and preprocessing

### AnalysisEngine (`core/analysis_engine.py`)
- Statistical calculations
- ROI management and analysis
- Saturation detection
- Histogram generation

### AttenuationCalculator (`core/attenuation_calc.py`)
- X-ray physics calculations
- Material attenuation coefficients (Al, Fe, Cu, Pb)
- Exposure time recommendations
- Filter optimization suggestions

### GUI (`gui/main_window.py`)
- Tkinter-based interface
- Interactive image viewer with ROI selection
- Real-time analysis visualization
- Results export functionality

## Example Analysis Report

```json
{
  "analysis_parameters": {
    "threshold": 10.0,
    "saturation_threshold": 65535,
    "num_rois": 2
  },
  "overall_statistics": {
    "overall": {
      "mean": 11.19,
      "median": 6.0,
      "max": 64042,
      "std": 321.27
    }
  },
  "saturation_analysis": {
    "saturation_percentage": 0.001,
    "total_saturated_pixels": 847
  },
  "attenuation_analysis": {
    "total_attenuation_factor": 1.02,
    "recommendations": {
      "exposure": {
        "exposure_time_s": 1.34,
        "reasoning": "Scaled to achieve target mean counts"
      }
    }
  }
}
```

## Configuration

Sample configuration files are provided in `examples/`:
- `sample_config.json`: Example analysis parameters and ROI definitions

## Testing

Run the test suite:
```bash
python3 test_hedm_analyzer.py  # Full test suite
python3 quick_test.py          # Quick functionality test
```

## Limitations

- GUI requires display (X11/Wayland) - use SSH with X forwarding for remote access
- Large datasets (>100 frames) may require significant memory
- Real-time EPICS PV support is planned but not yet implemented

## Future Enhancements

- Real-time data analysis with EPICS PV connections
- Advanced filtering and noise reduction
- 3D visualization capabilities
- Batch processing scripts
- Web-based interface option

## Technical Details

- **Language**: Python 3.7+
- **GUI Framework**: Tkinter
- **Scientific Computing**: NumPy, SciPy
- **Image Processing**: PIL/Pillow, scikit-image
- **Data I/O**: HDF5py
- **Visualization**: Matplotlib

## License

This software is provided for research and educational purposes.

## Contact

For questions or issues, please check the application logs (`hedm_analyzer.log`) for detailed error information.
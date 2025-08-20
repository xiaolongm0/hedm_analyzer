"""
Data handler for HEDM X-ray image analysis
Supports HDF5 files, image sequences, and future EPICS PV connections
"""

import numpy as np
import h5py
from PIL import Image
import os
import glob
from typing import List, Tuple, Optional, Union
import logging

class DataHandler:
    """Handle loading and preprocessing of HEDM detector data"""
    
    def __init__(self):
        self.data = None
        self.bright_field = None
        self.dark_field = None
        self.metadata = {}
        self.file_type = None
        self.shape = None
        self.dtype = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_hdf5(self, filepath: str) -> bool:
        """Load data from HDF5 file (preferred format)"""
        try:
            with h5py.File(filepath, 'r') as f:
                # Load projection data
                if 'exchange/data' in f:
                    self.data = f['exchange/data'][:]
                    self.logger.info(f"Loaded {self.data.shape[0]} projection images")
                else:
                    self.logger.error("No 'exchange/data' found in HDF5 file")
                    return False
                
                # Load bright/dark fields if available
                if 'exchange/bright' in f:
                    self.bright_field = f['exchange/bright'][0]
                if 'exchange/dark' in f:
                    self.dark_field = f['exchange/dark'][0]
                
                # Extract metadata
                self.metadata = self._extract_hdf5_metadata(f)
                
            self.file_type = 'hdf5'
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            
            self.logger.info(f"HDF5 data loaded: shape={self.shape}, dtype={self.dtype}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading HDF5 file: {e}")
            return False
    
    def load_image_sequence(self, directory: str, pattern: str = "*.png") -> bool:
        """Load sequence of image files"""
        try:
            # Find all matching files
            file_pattern = os.path.join(directory, pattern)
            files = sorted(glob.glob(file_pattern))
            
            if not files:
                self.logger.error(f"No files found matching {file_pattern}")
                return False
            
            # Load first image to get dimensions
            first_img = np.array(Image.open(files[0]))
            img_shape = first_img.shape
            
            # Initialize data array
            self.data = np.zeros((len(files), *img_shape), dtype=first_img.dtype)
            self.data[0] = first_img
            
            # Load remaining images
            for i, filepath in enumerate(files[1:], 1):
                img = np.array(Image.open(filepath))
                if img.shape != img_shape:
                    self.logger.warning(f"Image {filepath} has different shape: {img.shape}")
                    continue
                self.data[i] = img
            
            self.file_type = 'image_sequence'
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            
            # Create basic metadata
            self.metadata = {
                'num_images': len(files),
                'source_directory': directory,
                'pattern': pattern,
                'files': files
            }
            
            self.logger.info(f"Loaded {len(files)} images: shape={self.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading image sequence: {e}")
            return False
    
    def load_single_image(self, filepath: str) -> bool:
        """Load a single image file"""
        try:
            img = np.array(Image.open(filepath))
            self.data = img[np.newaxis, ...]  # Add frame dimension
            
            self.file_type = 'single_image'
            self.shape = self.data.shape
            self.dtype = self.data.dtype
            
            self.metadata = {
                'source_file': filepath,
                'original_shape': img.shape
            }
            
            self.logger.info(f"Single image loaded: shape={self.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading single image: {e}")
            return False
    
    def _extract_hdf5_metadata(self, h5file) -> dict:
        """Extract metadata from HDF5 file"""
        metadata = {}
        
        try:
            # Extract scan parameters
            if 'measurement/process/scan_parameters' in h5file:
                scan_params = h5file['measurement/process/scan_parameters']
                metadata['scan_parameters'] = {}
                
                for key in ['start', 'end', 'step', 'speed', 'steptime']:
                    if key in scan_params:
                        metadata['scan_parameters'][key] = float(scan_params[key][0])
            
            # Extract detector settings
            if 'WM' in h5file:
                wm_group = h5file['WM']
                metadata['detector_settings'] = {}
                
                # Common detector parameters
                params = ['AcqTime', 'Gain', 'Temperature', 'SizeX', 'SizeY', 'BinX', 'BinY']
                for param in params:
                    if param in wm_group:
                        values = wm_group[param][:]
                        if len(values) > 0:
                            metadata['detector_settings'][param] = float(values[0])
            
            # Extract timestamps
            if 'measurement/process' in h5file:
                proc_group = h5file['measurement/process']
                if 'start_date' in proc_group:
                    metadata['start_date'] = proc_group['start_date'][0].decode('utf-8')
                if 'end_date' in proc_group:
                    metadata['end_date'] = proc_group['end_date'][0].decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Could not extract all metadata: {e}")
        
        return metadata
    
    def apply_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply pixel mask to exclude certain pixels"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        if mask.shape != self.data.shape[-2:]:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {self.data.shape[-2:]}")
        
        # Create masked data (set masked pixels to NaN for statistics)
        masked_data = self.data.astype(np.float64)
        masked_data[:, ~mask] = np.nan
        
        return masked_data
    
    def apply_threshold(self, threshold: float) -> np.ndarray:
        """Apply intensity threshold to data"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        return self.data >= threshold
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get specific frame from data"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        if frame_idx >= self.data.shape[0]:
            raise IndexError(f"Frame index {frame_idx} out of range (max: {self.data.shape[0]-1})")
        
        return self.data[frame_idx]
    
    def get_roi_data(self, roi: Tuple[int, int, int, int], frame_idx: Optional[int] = None) -> np.ndarray:
        """Extract ROI data from specified region
        Args:
            roi: (x_start, y_start, x_end, y_end) in pixels
            frame_idx: specific frame, or None for all frames
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        x_start, y_start, x_end, y_end = roi
        
        if frame_idx is not None:
            return self.data[frame_idx, y_start:y_end, x_start:x_end]
        else:
            return self.data[:, y_start:y_end, x_start:x_end]
    
    def get_data_stats(self) -> dict:
        """Get basic statistics about loaded data"""
        if self.data is None:
            return {}
        
        return {
            'shape': self.shape,
            'dtype': str(self.dtype),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'total_pixels': int(np.prod(self.shape)),
            'file_type': self.file_type
        }
    
    def load_mask_file(self, filepath: str) -> np.ndarray:
        """Load mask from file (PNG, TIFF, or text)"""
        try:
            if filepath.lower().endswith(('.png', '.tiff', '.tif')):
                # Load as image
                mask_img = Image.open(filepath)
                mask = np.array(mask_img) > 0  # Convert to boolean
            else:
                # Try to load as text file
                mask = np.loadtxt(filepath, dtype=bool)
            
            self.logger.info(f"Mask loaded: shape={mask.shape}, {np.sum(mask)} valid pixels")
            return mask
            
        except Exception as e:
            self.logger.error(f"Error loading mask file: {e}")
            return None
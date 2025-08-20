"""
File utilities for HEDM analyzer
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Any
import logging

def save_results_json(results: Dict, filepath: str) -> bool:
    """Save analysis results to JSON file"""
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_numpy_to_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logging.info(f"Results saved to {filepath}")
        return True
    
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")
        return False

def save_statistics_csv(statistics: Dict, filepath: str) -> bool:
    """Save statistics to CSV file"""
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['ROI', 'Metric', 'Value'])
            
            # Write statistics
            for roi_name, stats in statistics.items():
                if isinstance(stats, dict):
                    for metric, value in stats.items():
                        if metric != 'percentiles':
                            writer.writerow([roi_name, metric, value])
                        else:
                            # Write percentiles separately
                            for pct, val in value.items():
                                writer.writerow([roi_name, f'percentile_{pct}', val])
        
        logging.info(f"Statistics saved to {filepath}")
        return True
    
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        return False

def convert_numpy_to_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj

def validate_file_path(filepath: str, create_dirs: bool = True) -> bool:
    """Validate file path and create directories if needed"""
    try:
        directory = os.path.dirname(filepath)
        
        if directory and not os.path.exists(directory):
            if create_dirs:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            else:
                logging.error(f"Directory does not exist: {directory}")
                return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error validating path: {e}")
        return False

def get_file_info(filepath: str) -> Dict:
    """Get basic file information"""
    try:
        stat = os.stat(filepath)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024*1024),
            'modified_time': stat.st_mtime,
            'extension': os.path.splitext(filepath)[1].lower()
        }
    except Exception as e:
        logging.error(f"Error getting file info: {e}")
        return {}
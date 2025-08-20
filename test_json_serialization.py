#!/usr/bin/env python3
"""
Test script to verify JSON serialization fix
"""

import json
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.attenuation_calc import AttenuationSettings, ScanConditions, AttenuationCalculator
from gui.main_window import CustomJSONEncoder

def test_json_serialization():
    """Test that AttenuationSettings objects can be serialized to JSON"""
    
    # Create test objects
    att_setting = AttenuationSettings('Al', 1.0, 'Test filter')
    scan_conditions = ScanConditions(
        energy_kev=80.0,
        exposure_time_s=0.5,
        attenuation_settings=[att_setting]
    )
    
    # Create test data that might be in analysis results
    test_data = {
        'test_attenuation_setting': att_setting,
        'test_scan_conditions': scan_conditions,
        'test_list': [att_setting, {'key': 'value'}],
        'test_dict': {
            'nested': att_setting,
            'regular': 'string'
        }
    }
    
    try:
        # Test serialization with custom encoder
        json_string = json.dumps(test_data, indent=2, cls=CustomJSONEncoder)
        print("✅ JSON serialization successful!")
        print("Serialized JSON:")
        print(json_string)
        
        # Test deserialization
        parsed_data = json.loads(json_string)
        print("\n✅ JSON deserialization successful!")
        print("Parsed data structure:")
        print(json.dumps(parsed_data, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing JSON serialization fix...")
    success = test_json_serialization()
    
    if success:
        print("\n🎉 All tests passed! The JSON serialization issue is fixed.")
    else:
        print("\n💥 Tests failed. The issue persists.")
        sys.exit(1)

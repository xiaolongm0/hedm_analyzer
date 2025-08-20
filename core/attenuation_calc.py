"""
X-ray attenuation calculations for HEDM analysis
Calculates attenuation factors and recommends scan parameters
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AttenuationSettings:
    """Attenuation filter settings"""
    material: str
    thickness_mm: float
    description: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'material': self.material,
            'thickness_mm': self.thickness_mm,
            'description': self.description
        }

@dataclass
class ScanConditions:
    """X-ray scan conditions"""
    energy_kev: float
    exposure_time_s: float
    attenuation_settings: list  # List of AttenuationSettings
    beam_current_ma: float = 100.0
    detector_distance_mm: float = 1000.0

class AttenuationCalculator:
    """Calculate X-ray attenuation and recommend scan parameters"""
    
    # Mass attenuation coefficients (cm²/g) at different energies
    # Data from NIST XCOM database
    ATTENUATION_DATA = {
        'Al': {  # Aluminum
            10.0: 5.329,   # 10 keV
            15.0: 1.582,   # 15 keV  
            20.0: 0.835,   # 20 keV
            30.0: 0.324,   # 30 keV
            50.0: 0.146,   # 50 keV
            80.0: 0.078,   # 80 keV
            100.0: 0.065   # 100 keV
        },
        'Fe': {  # Iron
            10.0: 23.94,
            15.0: 6.765,
            20.0: 3.441,
            30.0: 1.298,
            50.0: 0.569,
            80.0: 0.296,
            100.0: 0.243
        },
        'Cu': {  # Copper
            10.0: 27.38,
            15.0: 7.710,
            20.0: 3.910,
            30.0: 1.470,
            50.0: 0.638,
            80.0: 0.327,
            100.0: 0.267
        },
        'Pb': {  # Lead
            10.0: 121.9,
            15.0: 32.73,
            20.0: 15.99,
            30.0: 5.549,
            50.0: 2.235,
            80.0: 1.072,
            100.0: 0.845
        }
    }
    
    # Material densities (g/cm³)
    DENSITIES = {
        'Al': 2.70,
        'Fe': 7.87,
        'Cu': 8.96,
        'Pb': 11.34
    }
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def interpolate_attenuation_coefficient(self, material: str, energy_kev: float) -> float:
        """Interpolate mass attenuation coefficient for given material and energy"""
        if material not in self.ATTENUATION_DATA:
            raise ValueError(f"Material '{material}' not in database. Available: {list(self.ATTENUATION_DATA.keys())}")
        
        data = self.ATTENUATION_DATA[material]
        energies = np.array(list(data.keys()))
        coeffs = np.array(list(data.values()))
        
        # Interpolate (or extrapolate) in log space for better behavior
        log_energies = np.log(energies)
        log_coeffs = np.log(coeffs)
        log_energy = np.log(energy_kev)
        
        # Linear interpolation in log-log space
        log_coeff = np.interp(log_energy, log_energies, log_coeffs)
        
        return np.exp(log_coeff)
    
    def calculate_transmission(self, material: str, thickness_mm: float, energy_kev: float) -> float:
        """Calculate X-ray transmission through material"""
        # Get mass attenuation coefficient (cm²/g)
        mu_mass = self.interpolate_attenuation_coefficient(material, energy_kev)
        
        # Get density (g/cm³)
        density = self.DENSITIES[material]
        
        # Linear attenuation coefficient (cm⁻¹)
        mu_linear = mu_mass * density
        
        # Thickness in cm
        thickness_cm = thickness_mm / 10.0
        
        # Beer-Lambert law: I/I₀ = exp(-μt)
        transmission = np.exp(-mu_linear * thickness_cm)
        
        return transmission
    
    def calculate_attenuation_factor(self, scan_conditions: ScanConditions) -> float:
        """Calculate total attenuation factor from all filters"""
        total_transmission = 1.0
        
        for att_setting in scan_conditions.attenuation_settings:
            transmission = self.calculate_transmission(
                att_setting.material,
                att_setting.thickness_mm,
                scan_conditions.energy_kev
            )
            total_transmission *= transmission
        
        # Attenuation factor is reciprocal of transmission
        attenuation_factor = 1.0 / total_transmission
        
        return attenuation_factor
    
    def recommend_exposure_time(self, current_stats: Dict, scan_conditions: ScanConditions,
                              target_counts: int = 30000) -> Tuple[float, Dict]:
        """Recommend optimal exposure time based on current statistics"""
        current_mean = current_stats.get('mean', 0)
        current_max = current_stats.get('max', 0)
        current_exposure = scan_conditions.exposure_time_s
        
        recommendations = {}
        
        if current_mean == 0:
            recommendations['exposure_time_s'] = current_exposure * 2
            recommendations['reasoning'] = "No signal detected, doubling exposure time"
            recommendations['confidence'] = 'low'
            return recommendations['exposure_time_s'], recommendations
        
        # Calculate scaling factor to reach target counts
        scale_factor = target_counts / current_mean
        recommended_exposure = current_exposure * scale_factor
        
        # Check for saturation risk
        detector_max = 65535  # Assume 16-bit detector
        predicted_max = current_max * scale_factor
        
        if predicted_max > detector_max * 0.9:  # 90% of detector range
            # Reduce to avoid saturation
            safe_scale = (detector_max * 0.8) / current_max
            recommended_exposure = current_exposure * safe_scale
            recommendations['reasoning'] = f"Reduced to avoid saturation (predicted max: {predicted_max:.0f})"
            recommendations['confidence'] = 'high'
        else:
            recommendations['reasoning'] = f"Scaled to achieve target mean counts ({target_counts})"
            recommendations['confidence'] = 'medium'
        
        # Practical limits
        recommended_exposure = max(0.001, min(recommended_exposure, 60.0))  # 1ms to 60s
        
        recommendations.update({
            'exposure_time_s': recommended_exposure,
            'current_exposure_s': current_exposure,
            'scale_factor': scale_factor,
            'predicted_mean': current_mean * (recommended_exposure / current_exposure),
            'predicted_max': current_max * (recommended_exposure / current_exposure),
            'target_counts': target_counts
        })
        
        return recommended_exposure, recommendations
    
    def recommend_attenuation(self, current_stats: Dict, scan_conditions: ScanConditions,
                            target_counts: int = 30000) -> Dict:
        """Recommend attenuation changes based on current intensity"""
        current_mean = current_stats.get('mean', 0)
        current_max = current_stats.get('max', 0)
        
        recommendations = {
            'action': 'none',
            'reasoning': '',
            'suggested_filters': [],
            'confidence': 'medium'
        }
        
        detector_max = 65535
        
        # Check if beam is too intense (saturation or near-saturation)
        if current_max > detector_max * 0.8:
            # Need more attenuation
            current_factor = self.calculate_attenuation_factor(scan_conditions)
            needed_reduction = current_max / (detector_max * 0.6)  # Target 60% of detector range
            
            # Suggest additional aluminum filters
            additional_al_thickness = self._suggest_additional_attenuation(
                scan_conditions.energy_kev, needed_reduction, 'Al'
            )
            
            recommendations.update({
                'action': 'increase_attenuation',
                'reasoning': f'Current max ({current_max:.0f}) near saturation. Need {needed_reduction:.1f}x more attenuation',
                'suggested_filters': [AttenuationSettings('Al', additional_al_thickness, 'Additional filter').to_dict()],
                'confidence': 'high'
            })
            
        elif current_mean < target_counts * 0.1:
            # Signal too weak, might need less attenuation
            if len(scan_conditions.attenuation_settings) > 0:
                recommendations.update({
                    'action': 'decrease_attenuation',
                    'reasoning': f'Signal too weak (mean: {current_mean:.0f}). Consider removing filters',
                    'suggested_filters': [],
                    'confidence': 'medium'
                })
            else:
                recommendations.update({
                    'action': 'increase_exposure',
                    'reasoning': f'Signal weak but no filters to remove. Increase exposure time',
                    'confidence': 'high'
                })
        
        return recommendations
    
    def _suggest_additional_attenuation(self, energy_kev: float, reduction_factor: float, 
                                      material: str = 'Al') -> float:
        """Calculate thickness needed for additional attenuation"""
        # We want: exp(-μt) = 1/reduction_factor
        # So: t = -ln(1/reduction_factor) / μ = ln(reduction_factor) / μ
        
        mu_mass = self.interpolate_attenuation_coefficient(material, energy_kev)
        density = self.DENSITIES[material]
        mu_linear = mu_mass * density  # cm⁻¹
        
        thickness_cm = np.log(reduction_factor) / mu_linear
        thickness_mm = thickness_cm * 10.0
        
        # Round to practical values
        practical_thicknesses = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        thickness_mm = min(practical_thicknesses, key=lambda x: abs(x - thickness_mm))
        
        return thickness_mm
    
    def generate_attenuation_report(self, scan_conditions: ScanConditions, 
                                  current_stats: Optional[Dict] = None) -> Dict:
        """Generate comprehensive attenuation analysis report"""
        report = {
            'scan_conditions': {
                'energy_kev': scan_conditions.energy_kev,
                'exposure_time_s': scan_conditions.exposure_time_s,
                'beam_current_ma': scan_conditions.beam_current_ma,
                'detector_distance_mm': scan_conditions.detector_distance_mm
            },
            'attenuation_filters': [],
            'total_attenuation_factor': 0,
            'total_transmission': 0,
            'recommendations': {}
        }
        
        # Analyze each filter
        for att_setting in scan_conditions.attenuation_settings:
            transmission = self.calculate_transmission(
                att_setting.material,
                att_setting.thickness_mm,
                scan_conditions.energy_kev
            )
            
            filter_info = {
                'material': att_setting.material,
                'thickness_mm': att_setting.thickness_mm,
                'description': att_setting.description,
                'transmission': transmission,
                'attenuation_factor': 1.0 / transmission,
                'attenuation_db': -20 * np.log10(transmission)  # in decibels
            }
            report['attenuation_filters'].append(filter_info)
        
        # Calculate total attenuation
        total_factor = self.calculate_attenuation_factor(scan_conditions)
        report['total_attenuation_factor'] = total_factor
        report['total_transmission'] = 1.0 / total_factor
        
        # Add recommendations if current statistics provided
        if current_stats:
            exposure_rec, exposure_details = self.recommend_exposure_time(current_stats, scan_conditions=scan_conditions)
            attenuation_rec = self.recommend_attenuation(current_stats, scan_conditions)
            
            report['recommendations'] = {
                'exposure': exposure_details,
                'attenuation': attenuation_rec
            }
        
        return report
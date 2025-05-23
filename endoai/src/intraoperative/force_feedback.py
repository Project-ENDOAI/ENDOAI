"""
Module for processing force and tactile sensor data in intraoperative endometriosis surgery.

This module processes data from force feedback sensors, tip pressure sensors, and haptic
feedback systems to provide real-time tissue resistance information and safety alerts.
It includes tools for tissue characterization and haptic feedback generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import threading


class ForceSensorProcessor:
    """
    Processor for force sensor data to detect tissue resistance and provide guidance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the force sensor processor.
        
        Args:
            config: Configuration parameters for force processing
        """
        self.config = config
        self.force_threshold = config.get("force_threshold", 5.0)  # N
        self.calibration_factor = config.get("calibration_factor", 1.0)
        self.safety_threshold = config.get("safety_threshold", 8.0)  # N
        self.history = []
        self.history_max_size = config.get("history_size", 100)
    
    def process_reading(self, force_data: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Process a force sensor reading.
        
        Args:
            force_data: Array of force measurements
            timestamp: Timestamp of the measurement
            
        Returns:
            Dictionary with processed results and safety alerts
        """
        # Apply calibration
        calibrated_data = force_data * self.calibration_factor
        
        # Calculate magnitude if force is multi-dimensional
        if calibrated_data.ndim > 0 and calibrated_data.size > 1:
            magnitude = np.linalg.norm(calibrated_data)
        else:
            magnitude = float(calibrated_data)
        
        # Update history
        self.history.append((timestamp, magnitude))
        if len(self.history) > self.history_max_size:
            self.history.pop(0)
        
        # Compute metrics
        gradient = self._compute_gradient() if len(self.history) > 1 else 0.0
        
        # Generate results
        result = {
            "force_magnitude": magnitude,
            "gradient": gradient,
            "exceeded_threshold": magnitude > self.force_threshold,
            "safety_alert": magnitude > self.safety_threshold,
            "timestamp": timestamp
        }
        
        # Add tissue characterization if enabled
        if self.config.get("characterize_tissue", False):
            result["tissue_characterization"] = self._characterize_tissue(magnitude, gradient)
        
        return result
    
    def _compute_gradient(self) -> float:
        """
        Compute the rate of change of force magnitude.
        
        Returns:
            Force gradient (N/s)
        """
        if len(self.history) < 2:
            return 0.0
        
        # Get the last two readings
        (t1, f1), (t2, f2) = self.history[-2], self.history[-1]
        
        # Calculate gradient
        dt = t2 - t1
        if dt > 0:
            return (f2 - f1) / dt
        return 0.0
    
    def _characterize_tissue(self, magnitude: float, gradient: float) -> str:
        """
        Characterize tissue based on force magnitude and gradient.
        
        Args:
            magnitude: Force magnitude
            gradient: Force gradient
            
        Returns:
            Tissue characterization label
        """
        # Very simple characterization based on force thresholds
        if magnitude < 2.0:
            return "soft_tissue"
        elif magnitude < 5.0:
            return "normal_tissue"
        elif magnitude < 8.0:
            return "fibrotic_tissue"
        else:
            return "hard_tissue"
    
    def calibrate(self, calibration_readings: List[Tuple[np.ndarray, float]]) -> float:
        """
        Calibrate the force sensor using known reference forces.
        
        Args:
            calibration_readings: List of (measured_force, reference_force) tuples
            
        Returns:
            Calibration factor
        """
        if not calibration_readings:
            return self.calibration_factor
        
        # Extract measured and reference values
        measured = np.array([m for m, _ in calibration_readings])
        reference = np.array([r for _, r in calibration_readings])
        
        # Compute calibration factor (simple linear scaling)
        if np.sum(measured) > 0:
            self.calibration_factor = np.mean(reference / measured)
        
        return self.calibration_factor


class TactileFeedbackProcessor:
    """
    Processor for tactile feedback data from haptic sensors or tip pressure sensors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactile feedback processor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.pressure_threshold = config.get("pressure_threshold", 100.0)  # kPa
        self.contact_threshold = config.get("contact_threshold", 20.0)  # kPa
        self.spatial_resolution = config.get("spatial_resolution", (16, 16))  # sensor array size
    
    def process_tactile_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a tactile sensor frame (pressure distribution).
        
        Args:
            frame: 2D array of pressure values across the tactile sensor
            
        Returns:
            Dictionary with processed results
        """
        # Ensure frame has the right dimensions
        if frame.ndim == 1:
            # Reshape 1D array to 2D based on spatial resolution
            frame = frame.reshape(self.spatial_resolution)
        
        # Calculate key metrics
        max_pressure = np.max(frame)
        contact_points = np.sum(frame > self.contact_threshold)
        pressure_centroid = self._compute_pressure_centroid(frame)
        
        # Generate contact pattern representation
        contact_pattern = (frame > self.contact_threshold).astype(np.uint8)
        
        return {
            "max_pressure": max_pressure,
            "contact_points": contact_points,
            "pressure_centroid": pressure_centroid,
            "contact_pattern": contact_pattern,
            "high_pressure_alert": max_pressure > self.pressure_threshold
        }
    
    def _compute_pressure_centroid(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Compute the centroid of pressure distribution.
        
        Args:
            frame: 2D array of pressure values
            
        Returns:
            (x, y) coordinates of pressure centroid
        """
        # Avoid division by zero
        total = np.sum(frame)
        if total == 0:
            return (0.0, 0.0)
        
        # Create coordinate grids
        h, w = frame.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Compute weighted average of coordinates
        cx = np.sum(x * frame) / total
        cy = np.sum(y * frame) / total
        
        return (cx, cy)
    
    def analyze_tissue_properties(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze tissue properties based on tactile feedback.
        
        Args:
            frame: 2D array of pressure values
            
        Returns:
            Dictionary of estimated tissue properties
        """
        # Calculate simple metrics that correlate with tissue properties
        if np.max(frame) == 0:
            return {
                "stiffness": 0.0,
                "elasticity": 0.0,
                "homogeneity": 1.0  # Uniform zero pressure is perfectly homogeneous
            }
        
        # Normalize the frame
        norm_frame = frame / np.max(frame)
        
        # Stiffness correlates with pressure for a given displacement
        stiffness = np.mean(norm_frame)
        
        # Homogeneity: how uniform is the pressure distribution
        # Higher value means more homogeneous (smooth) distribution
        homogeneity = 1.0 - np.std(norm_frame)
        
        # Elasticity is a placeholder - would require temporal data
        elasticity = 0.5  # Placeholder
        
        return {
            "stiffness": float(stiffness),
            "elasticity": float(elasticity),
            "homogeneity": float(homogeneity)
        }


class InstrumentLoadMonitor:
    """
    Monitor for instrument shaft load sensors that detect torque, bending, and tension.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the instrument load monitor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.torque_threshold = config.get("torque_threshold", 2.0)  # Nm
        self.bending_threshold = config.get("bending_threshold", 10.0)  # degrees
        self.tension_threshold = config.get("tension_threshold", 50.0)  # N
        
        # Initialize history for each metric
        self.torque_history = []
        self.bending_history = []
        self.tension_history = []
        self.history_max_size = config.get("history_size", 100)
    
    def process_load_data(self, 
                         torque: float, 
                         bending: float, 
                         tension: float, 
                         timestamp: float) -> Dict[str, Any]:
        """
        Process instrument load sensor data.
        
        Args:
            torque: Measured torque (Nm)
            bending: Measured bending (degrees)
            tension: Measured tension (N)
            timestamp: Measurement timestamp
            
        Returns:
            Dictionary with processed results and alerts
        """
        # Update history
        self.torque_history.append((timestamp, torque))
        self.bending_history.append((timestamp, bending))
        self.tension_history.append((timestamp, tension))
        
        # Trim history to maximum size
        if len(self.torque_history) > self.history_max_size:
            self.torque_history.pop(0)
        if len(self.bending_history) > self.history_max_size:
            self.bending_history.pop(0)
        if len(self.tension_history) > self.history_max_size:
            self.tension_history.pop(0)
        
        # Check thresholds
        torque_alert = torque > self.torque_threshold
        bending_alert = bending > self.bending_threshold
        tension_alert = tension > self.tension_threshold
        
        # Generate result
        result = {
            "torque": torque,
            "bending": bending,
            "tension": tension,
            "timestamp": timestamp,
            "alerts": {
                "torque_exceeded": torque_alert,
                "bending_exceeded": bending_alert,
                "tension_exceeded": tension_alert
            },
            "safety_alert": torque_alert or bending_alert or tension_alert
        }
        
        # Add instrument stress analysis if enabled
        if self.config.get("stress_analysis", True):
            result["stress_analysis"] = self._analyze_instrument_stress(torque, bending, tension)
        
        return result
    
    def _analyze_instrument_stress(self, torque: float, bending: float, tension: float) -> str:
        """
        Analyze instrument stress level based on combined load metrics.
        
        Args:
            torque: Measured torque
            bending: Measured bending
            tension: Measured tension
            
        Returns:
            Stress level category
        """
        # Normalize each metric relative to its threshold
        norm_torque = torque / self.torque_threshold
        norm_bending = bending / self.bending_threshold
        norm_tension = tension / self.tension_threshold
        
        # Combine for overall stress score
        combined_stress = 0.4 * norm_torque + 0.3 * norm_bending + 0.3 * norm_tension
        
        # Categorize stress level
        if combined_stress < 0.5:
            return "low"
        elif combined_stress < 0.8:
            return "moderate"
        elif combined_stress < 1.0:
            return "high"
        else:
            return "critical"


class HapticFeedbackManager:
    """
    Manager for generating haptic feedback based on sensor data.
    This would interface with haptic devices to simulate tactile sensations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the haptic feedback manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.feedback_enabled = config.get("enabled", True)
        self.intensity_scaling = config.get("intensity_scaling", 1.0)
        self.feedback_mode = config.get("mode", "continuous")
        
        # Initialize feedback thread if in continuous mode
        self.feedback_thread = None
        self.stop_thread = False
        if self.feedback_mode == "continuous" and self.feedback_enabled:
            self.feedback_thread = threading.Thread(target=self._continuous_feedback_loop)
            self.feedback_thread.daemon = True
            self.feedback_thread.start()
    
    def generate_feedback(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate haptic feedback signals based on sensor data.
        
        Args:
            sensor_data: Combined sensor data including force, visual, etc.
            
        Returns:
            Dictionary with feedback signals
        """
        if not self.feedback_enabled:
            return {"status": "disabled"}
        
        # Extract relevant data from sensor inputs
        force_magnitude = sensor_data.get("force", {}).get("force_magnitude", 0.0)
        tissue_type = sensor_data.get("force", {}).get("tissue_characterization", "unknown")
        safety_alert = any(data.get("safety_alert", False) for data in sensor_data.values())
        
        # Calculate feedback parameters
        intensity = self._calculate_feedback_intensity(force_magnitude)
        pattern = self._select_feedback_pattern(tissue_type, safety_alert)
        frequency = self._calculate_feedback_frequency(tissue_type)
        
        return {
            "intensity": intensity,
            "pattern": pattern,
            "frequency": frequency,
            "status": "active",
            "alert_mode": safety_alert
        }
    
    def _calculate_feedback_intensity(self, force: float) -> float:
        """
        Calculate feedback intensity based on force.
        
        Args:
            force: Force magnitude
            
        Returns:
            Feedback intensity (0-1)
        """
        # Scale force to intensity with configurable scaling
        raw_intensity = force * self.intensity_scaling
        
        # Clamp to [0, 1] range
        return min(max(raw_intensity, 0.0), 1.0)
    
    def _select_feedback_pattern(self, tissue_type: str, alert: bool) -> str:
        """
        Select appropriate haptic feedback pattern.
        
        Args:
            tissue_type: Identified tissue type
            alert: Whether safety alert is active
            
        Returns:
            Feedback pattern identifier
        """
        if alert:
            return "pulse_strong"
        
        # Different patterns for different tissues
        pattern_map = {
            "soft_tissue": "smooth",
            "normal_tissue": "textured",
            "fibrotic_tissue": "rough",
            "hard_tissue": "solid",
            "unknown": "neutral"
        }
        
        return pattern_map.get(tissue_type, "neutral")
    
    def _calculate_feedback_frequency(self, tissue_type: str) -> float:
        """
        Calculate feedback frequency based on tissue type.
        
        Args:
            tissue_type: Identified tissue type
            
        Returns:
            Feedback frequency (Hz)
        """
        # Different frequencies for different tissues
        frequency_map = {
            "soft_tissue": 20.0,
            "normal_tissue": 40.0,
            "fibrotic_tissue": 60.0,
            "hard_tissue": 80.0,
            "unknown": 30.0
        }
        
        return frequency_map.get(tissue_type, 30.0)
    
    def _continuous_feedback_loop(self):
        """Background thread for continuous haptic feedback updates."""
        while not self.stop_thread:
            # This would interface with actual haptic hardware
            # For now, just sleep to simulate the update rate
            time.sleep(0.05)  # 20Hz update rate
    
    def stop(self):
        """Stop the continuous feedback thread."""
        if self.feedback_thread:
            self.stop_thread = True
            self.feedback_thread.join(timeout=1.0)


# Example usage
if __name__ == "__main__":
    # Example of ForceSensorProcessor
    force_config = {"force_threshold": 5.0, "characterize_tissue": True}
    force_processor = ForceSensorProcessor(force_config)
    
    # Process a simulated force reading
    force_reading = np.array([3.2, 1.5, 0.8])  # X, Y, Z components
    result = force_processor.process_reading(force_reading, time.time())
    
    print(f"Force magnitude: {result['force_magnitude']:.2f} N")
    print(f"Tissue characterization: {result.get('tissue_characterization', 'none')}")
    print(f"Safety alert: {'Yes' if result['safety_alert'] else 'No'}")

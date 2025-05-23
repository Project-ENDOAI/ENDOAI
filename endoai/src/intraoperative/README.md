# intraoperative

This folder contains code and resources for **intraoperative (real-time) guidance** in the ENDOAI project.

## Purpose

- Real-time segmentation and analysis of surgical video or imaging data.
- Tools and models for intraoperative lesion detection, organ identification, and risk mapping.
- Utilities for integrating AI outputs into surgical workflows.
- Integration of preoperative imaging data with real-time sensor feedback for comprehensive surgical guidance.

## Structure

- `video_segmentation.py` — Scripts and models for video-based segmentation.
- `lesion_detection.py` — Real-time lesion detection algorithms.
- `sensor_fusion.py` — Integrates data from multiple sensors for comprehensive analysis.
- `visual_processing.py` — Processing and enhancement of visual sensor data.
- `force_feedback.py` — Processing of force and tactile sensor data for haptic feedback.
- `preop_fusion.py` — Combines preoperative imaging data with real-time intraoperative sensor data.
- Other scripts and modules related to intraoperative support.

## Sensor Integration

The intraoperative module supports multiple sensor types used in endometriosis surgery:

### Visual Sensors
- CMOS/CCD Image Sensors from laparoscopes
- Stereoscopic Cameras for 3D vision
- Infrared Fluorescence Sensors for ICG dye visualization

### Force and Tactile Sensors
- Force Feedback Sensors for tissue resistance detection
- Tip Pressure Sensors for contact pressure monitoring
- Haptic Feedback Systems for surgeon guidance

### Other Sensors
- Motion and Position Sensors for instrument tracking
- Thermal Sensors for energy application monitoring
- Environmental Sensors for pneumoperitoneum control

## Preoperative-to-Intraoperative Integration

While most sensors operate during surgery, some provide integration with preoperative data:

- **Navigation & Imaging Fusion**: Systems like optical tracking markers and magnetic field trackers align preoperative MRI/CT data with real-time surgical views, bridging planning and execution
- **AI-Based Computer Vision**: Our algorithms incorporate preoperative image analysis to improve real-time lesion detection and risk assessment
- **Preoperative Registration**: Tools for registering diagnostic imaging (MRI, CT, ultrasound) to the intraoperative coordinate system

This integration ensures continuity between preoperative planning and intraoperative execution, allowing surgeons to reference critical information identified during planning while operating.

## Usage

Import modules as needed in your workflow or run scripts directly for real-time inference and guidance.

```python
# Example: Using the sensor fusion pipeline
from endoai.src.intraoperative.sensor_fusion import SensorFusionPipeline

# Initialize the pipeline
config = {...}  # Configuration parameters
pipeline = SensorFusionPipeline(config)

# Register sensors
pipeline.register_sensor("camera_1", "visual", {"resolution": (1920, 1080)})
pipeline.register_sensor("force_1", "force", {"range": (0, 10)})

# Update with sensor data
pipeline.update_sensor_data("camera_1", frame_data, timestamp)
pipeline.update_sensor_data("force_1", force_data, timestamp)

# Perform fusion and get results
results = pipeline.perform_fusion()
```

## Contributing

- Add new scripts or modules relevant to intraoperative guidance.
- Document all new files and functions.
- Follow the project's [coding standards](../../../COPILOT.md).

## See Also

- [../preoperative/](../preoperative/) — For preoperative planning and analysis.
- [../../README.md](../../README.md) — Project overview.

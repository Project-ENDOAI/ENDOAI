"""
License: MIT License
Author: Kevin
Version: 1.0.0
Change Log:
    - 1.0.0: Initial script for visualizing medical images and results.
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    print("Warning: Plotly is not installed. 3D visualization will not work.")

def visualize(image: np.ndarray, title: str = "Image") -> None:
    """
    Visualizes a medical image using matplotlib.

    Args:
        image (np.ndarray): Image to visualize.
        title (str): Title of the visualization.
    """
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_2d(lesion_slice: np.ndarray, risk_slice: np.ndarray) -> None:
    """
    Visualize 2D annotated map of lesions and risk zones.

    Args:
        lesion_slice (np.ndarray): 2D array representing lesion areas.
        risk_slice (np.ndarray): 2D array representing risk zones.
    """
    plt.imshow(lesion_slice, cmap="Reds", alpha=0.5)
    plt.imshow(risk_slice, cmap="Blues", alpha=0.3)
    plt.title("Lesion and Risk Zones")
    plt.show()

def visualize_3d(segmented_data: np.ndarray, risk_mask: np.ndarray) -> None:
    """
    Visualize 3D segmented data and risk zones.

    Args:
        segmented_data (np.ndarray): 3D array representing segmented lesion data.
        risk_mask (np.ndarray): 3D array representing risk zones.
    """
    if go is None:
        print("Error: Plotly is required for 3D visualization. Please install it using 'pip install plotly'.")
        return

    x, y, z = np.where(segmented_data > 0.5)
    x_r, y_r, z_r = np.where(risk_mask)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='red'), name="Lesions"))
    fig.add_trace(go.Scatter3d(x=x_r, y=y_r, z=z_r, mode='markers', marker=dict(size=2, color='blue'), name="Risk Zones"))
    fig.show()

# Example usage
# visualize_2d(lesion_slice, risk_slice)
# visualize_3d(segmented_data, risk_mask)

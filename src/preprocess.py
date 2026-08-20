import cv2
import numpy as np
import math

def bgr_to_cielab(image_bgr: np.ndarray) -> np.ndarray:
    """
    Converts standard OpenCV BGR image array to standard CIE L*a*b* space.
    OpenCV scales: L* -> [0, 255], a* -> [0, 255], b* -> [0, 255]
    where true a*=128 is neutral zero.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

def calculate_ita(l_val: float, b_val: float) -> float:
    """
    Calculates Individual Typology Angle (ITA) in degrees.
    Formula: ITA = (arctan((L* - 50) / b*)) * (180 / pi)
    
    Standard Standardized Scale:
      > 55°: Very Light
      41° to 55°: Light
      28° to 41°: Intermediate
      10° to 28°: Tan / Kayumanggi
      -30° to 10°: Brown / Dark
    """
    # Prevent division by zero if b* is exactly 0
    if b_val == 0:
        b_val = 0.0001
    
    ita_rad = math.atan((l_val - 50.0) / b_val)
    ita_deg = (ita_rad * 180.0) / math.pi
    return ita_deg

def extract_tissue_color_statistics(roi_lab: np.ndarray) -> dict:
    """
    Extracts mean and standard deviation across L*, a*, and b* channels
    for a given Region of Interest (ROI).
    """
    # Split the channels
    l_channel, a_channel, b_channel = cv2.split(roi_lab)
    
    # Scale OpenCV 8-bit LAB values back to standard scientific CIE scales
    # L*: 0 to 100, a*: -128 to +127, b*: -128 to +127
    l_std_scale = l_channel.astype(np.float32) * (100.0 / 255.0)
    a_std_scale = a_channel.astype(np.float32) - 128.0
    b_std_scale = b_channel.astype(np.float32) - 128.0

    return {
        "mean_L": float(np.mean(l_std_scale)),
        "std_L": float(np.std(l_std_scale)),
        "mean_a": float(np.mean(a_std_scale)),
        "std_a": float(np.std(a_std_scale)),
        "mean_b": float(np.mean(b_std_scale)),
        "std_b": float(np.std(b_std_scale)),
    }
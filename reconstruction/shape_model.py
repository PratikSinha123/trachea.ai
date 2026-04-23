"""Statistical shape model for trachea based on anatomical priors.

Encodes normal trachea geometry: typical diameters, tapering, cross-section
shape, and provides methods to generate idealized trachea volumes.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


# Anatomical reference values (adults, in mm)
TRACHEA_LENGTH_RANGE = (90, 130)       # 9-13 cm
TRACHEA_DIAMETER_UPPER = (18, 25)      # Near larynx
TRACHEA_DIAMETER_LOWER = (14, 20)      # Near carina
TRACHEA_WALL_THICKNESS = (2, 4)        # Wall thickness
TRACHEA_TAPER_RATIO = 0.85             # Lower/upper diameter ratio


class TracheaShapeModel:
    """Anatomical shape model for healthy trachea.

    Uses the centerline and cross-section measurements from the segmentation
    to build a smooth, idealized trachea shape.
    """

    def __init__(self):
        self.centerline = None
        self.diameters = None
        self.areas = None
        self.z_coords = None

    def fit(self, centerline, cross_sections):
        """Fit the shape model to observed trachea measurements.

        Identifies healthy regions and builds a smooth model.
        """
        if len(cross_sections) < 3:
            return False

        self.z_coords = np.array([cs["z_physical"] for cs in cross_sections])
        self.diameters = np.array([cs["equiv_diameter_mm"] for cs in cross_sections])
        self.areas = np.array([cs["area_mm2"] for cs in cross_sections])
        self.centerline = centerline

        return True

    def detect_anomalies(self, sensitivity=1.5):
        """Detect anomalous cross-sections that deviate from expected shape.

        Returns indices of anomalous slices and anomaly descriptions.
        """
        if self.diameters is None or len(self.diameters) < 5:
            return [], []

        # Smooth the diameter profile to get expected trend
        smooth_d = gaussian_filter1d(self.diameters, sigma=5)

        # Calculate residuals
        residuals = self.diameters - smooth_d
        std = np.std(residuals)

        anomalies = []
        descriptions = []

        for i, (d, sd, r) in enumerate(zip(self.diameters, smooth_d, residuals)):
            if abs(r) > sensitivity * std:
                anomalies.append(i)
                if r < 0:
                    severity = abs(r) / sd * 100
                    descriptions.append({
                        "index": i,
                        "z_mm": float(self.z_coords[i]),
                        "type": "stenosis",
                        "observed_mm": float(d),
                        "expected_mm": float(sd),
                        "deviation_pct": float(severity),
                    })
                else:
                    severity = r / sd * 100
                    descriptions.append({
                        "index": i,
                        "z_mm": float(self.z_coords[i]),
                        "type": "dilation",
                        "observed_mm": float(d),
                        "expected_mm": float(sd),
                        "deviation_pct": float(severity),
                    })

        return anomalies, descriptions

    def predict_healthy_profile(self):
        """Generate the predicted healthy diameter profile.

        Uses robust statistics from non-anomalous regions and smooth
        interpolation to create an idealized profile.
        """
        if self.diameters is None or len(self.diameters) < 3:
            return None, None

        anomaly_idx, _ = self.detect_anomalies()
        anomaly_set = set(anomaly_idx)

        # Get healthy measurements
        healthy_z = []
        healthy_d = []
        for i in range(len(self.diameters)):
            if i not in anomaly_set:
                healthy_z.append(self.z_coords[i])
                healthy_d.append(self.diameters[i])

        if len(healthy_z) < 3:
            # Not enough healthy data, use median
            med_d = np.median(self.diameters)
            return self.z_coords, np.full_like(self.z_coords, med_d)

        healthy_z = np.array(healthy_z)
        healthy_d = np.array(healthy_d)

        # Fit a smooth spline through healthy regions
        # Use smoothing factor proportional to data length
        s = len(healthy_z) * 0.5
        k = min(3, len(healthy_z) - 1)

        try:
            spline = UnivariateSpline(healthy_z, healthy_d, k=k, s=s)
            predicted_d = spline(self.z_coords)
        except Exception:
            # Fallback: simple linear interpolation
            predicted_d = np.interp(self.z_coords, healthy_z, healthy_d)

        # Clamp to anatomically plausible range
        predicted_d = np.clip(predicted_d, 8.0, 30.0)

        # Apply gentle Gaussian smoothing
        predicted_d = gaussian_filter1d(predicted_d, sigma=3)

        return self.z_coords, predicted_d

    def predict_healthy_centerline(self):
        """Generate a smoothed healthy centerline."""
        if self.centerline is None or len(self.centerline) < 3:
            return self.centerline

        smoothed = np.copy(self.centerline)
        for dim in range(3):
            smoothed[:, dim] = gaussian_filter1d(self.centerline[:, dim], sigma=3)

        return smoothed

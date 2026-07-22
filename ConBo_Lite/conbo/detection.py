"""Ellipse-based center and radius detection on normalized ConBo responses."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2 as cv
import numpy as np


@dataclass
class Detection:
    center: Tuple[float, float]
    axes: Tuple[float, float]
    angle: float
    score: float
    method: str

    @property
    def radius(self) -> float:
        return 0.25 * (self.axes[0] + self.axes[1])

    @property
    def circularity(self) -> float:
        return min(self.axes) / max(max(self.axes), 1e-9)


class EllipseDetector:
    """Reuse one EdgeDrawing detector and handle empty responses safely."""
    def __init__(self):
        self.edge = None
        if hasattr(cv, "ximgproc") and hasattr(cv.ximgproc, "createEdgeDrawing"):
            self.edge = cv.ximgproc.createEdgeDrawing()
            params = cv.ximgproc_EdgeDrawing_Params()
            params.MinPathLength = 5
            params.MinLineLength = 5
            params.PFmode = False
            params.NFAValidation = False
            self.edge.setParams(params)

    def detect(self, response: np.ndarray, hint: Optional[Tuple[float, float]] = None):
        """Return the plausible ellipse nearest the predicted ROI-local center."""
        if response is None or response.size == 0 or response.max() == 0:
            return None
        hint = hint or ((response.shape[1]-1)/2, (response.shape[0]-1)/2)
        candidates = []
        if self.edge is not None:
            self.edge.detectEdges(response)
            found = self.edge.detectEllipses()
            if found is not None:
                for e in found:
                    v = np.asarray(e).reshape(-1)
                    if len(v) >= 5:
                        cx, cy, a, b, angle = map(float, v[:5])
                        if a > 1 and b > 1:
                            candidates.append(Detection((cx, cy), (2*a, 2*b), angle, 0.0, "EdgeDrawing"))
        if not candidates:
            # Explicit geometric fallback for builds lacking usable EdgeDrawing output.
            level = max(1, int(response.max() * 0.35))
            contours, _ = cv.findContours((response >= level).astype(np.uint8),
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) >= 5:
                    (cx, cy), axes, angle = cv.fitEllipse(contour)
                    candidates.append(Detection((cx, cy), tuple(map(float, axes)),
                                                float(angle), 0.0, "fitEllipse"))
        if not candidates:
            return None
        scale = max(response.shape)
        for c in candidates:
            distance = np.hypot(c.center[0]-hint[0], c.center[1]-hint[1]) / scale
            circularity = min(c.axes) / max(c.axes)
            c.score = float(circularity - distance)
        return max(candidates, key=lambda c: c.score)

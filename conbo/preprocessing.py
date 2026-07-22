"""Batched MVGC, morphology, ROI extraction, and cached convolution."""

from dataclasses import dataclass
from typing import Sequence, Tuple
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from .config import ConBoConfig


@dataclass
class PreparedROI:
    raw_gray: np.ndarray
    isolated: np.ndarray
    response: np.ndarray
    bbox: Tuple[int, int, int, int]
    anchor: Tuple[float, float]


def extract_padded_roi(frame: np.ndarray, center: Tuple[float, float], size: int):
    """Extract a fixed-size ROI with zero padding and a valid full-frame bbox."""
    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    left = size // 2
    x0, y0 = cx - left, cy - left
    x1, y1 = x0 + size, y0 + size
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    roi = np.zeros((size, size, 3), dtype=frame.dtype)
    roi[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = frame[sy0:sy1, sx0:sx1]
    return roi, (sx0, sy0, sx1, sy1), (float(x0), float(y0))


def make_kernel(kind: str, size: int, radius_ratio: float, gaussian_sigma: float):
    """Create a normalized circular raised-cosine or Gaussian kernel."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2.0
    rr = np.hypot(xx-c, yy-c)
    if kind == "circular":
        radius = max(1.0, radius_ratio * size)
        kernel = 0.5 * (1.0 + np.cos(np.pi * np.minimum(rr / radius, 1.0)))
        kernel[rr > radius] = 0.0
    elif kind == "gaussian":
        sigma = max(float(gaussian_sigma), 1e-3)
        kernel = np.exp(-(rr ** 2) / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel.astype(np.float32)


class Convolver:
    """Use fast OpenCV on CPU and one batched PyTorch launch on CUDA."""
    def __init__(self, config: ConBoConfig):
        self.config = config
        self.requested_device = config.device
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device("cuda" if config.device == "cuda" else "cpu")
        self._kernels_np = {
            kind: make_kernel(kind, config.kernel_size, config.circular_radius_ratio,
                              config.gaussian_sigma) for kind in ("circular", "gaussian")
        }
        c = (config.kernel_size - 1) / 2.0
        axis = np.arange(config.kernel_size, dtype=np.float32) - c
        gaussian_1d = np.exp(-(axis ** 2) / (2.0 * max(config.gaussian_sigma, 1e-3) ** 2))
        self._gaussian_1d = (gaussian_1d / gaussian_1d.sum()).astype(np.float32)
        self._kernels_torch = {}
        morph_shapes = {"rect": cv.MORPH_RECT, "ellipse": cv.MORPH_ELLIPSE,
                        "cross": cv.MORPH_CROSS}
        self._morph_kernel = cv.getStructuringElement(
            morph_shapes[config.morphology_shape],
            (config.morphology_size, config.morphology_size))

    def configure(self, target_count: int, mode: str) -> torch.device:
        """Choose CUDA only when a batched workload can amortize transfer overhead."""
        if self.requested_device == "auto":
            multiplier = 2 if mode == "mag" else 1
            pixels = target_count * self.config.roi_size ** 2 * multiplier
            use_cuda = torch.cuda.is_available() and pixels >= self.config.gpu_min_batch_pixels
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device(self.requested_device)
        return self.device

    def kernel(self, kind: str):
        key = (kind, str(self.device))
        if key not in self._kernels_torch:
            self._kernels_torch[key] = torch.from_numpy(
                self._kernels_np[kind])[None, None].to(self.device)
        return self._kernels_torch[key]

    def _isolate_cpu(self, roi: np.ndarray):
        minimum = roi.min(axis=2).astype(np.float32)
        divider = max(float(minimum.max()), 1.0)
        corrected = np.power(minimum / divider, self.config.gamma) * divider
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        isolated = np.where(corrected > self.config.threshold, gray, 0).astype(np.uint8)
        return cv.morphologyEx(isolated, cv.MORPH_OPEN, self._morph_kernel)

    def _convolve_cpu(self, isolated: np.ndarray, kind: str):
        source = isolated.astype(np.float32)
        size = self.config.kernel_size
        before, after = size // 2, size - 1 - size // 2
        source = cv.copyMakeBorder(source, before, after, before, after,
                                   cv.BORDER_CONSTANT, value=0)
        if kind == "gaussian":
            response = cv.sepFilter2D(source, cv.CV_32F, self._gaussian_1d,
                                      self._gaussian_1d, borderType=cv.BORDER_CONSTANT)
        else:
            # The radial raised-cosine kernel is not separable. OpenCV selects
            # its optimized direct/DFT implementation without changing the kernel.
            response = cv.filter2D(source, cv.CV_32F, self._kernels_np[kind],
                                   borderType=cv.BORDER_CONSTANT)
        peak = float(response.max())
        if peak > 0:
            response *= 255.0 / peak
        return np.clip(response, 0, 255).astype(np.uint8)

    def _isolate_gpu(self, rois: np.ndarray):
        x = torch.from_numpy(rois).to(self.device, dtype=torch.float32)
        minimum = x.amin(dim=3)
        divider = minimum.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        corrected = torch.pow(minimum / divider, self.config.gamma) * divider
        gray = x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
        isolated = torch.where(corrected > self.config.threshold, gray, 0.0)[:, None]
        k = self.config.morphology_size
        eroded = -F.max_pool2d(-isolated, k, stride=1, padding=k//2)
        return F.max_pool2d(eroded, k, stride=1, padding=k//2)

    def _convolve_gpu(self, isolated: torch.Tensor, kind: str):
        if kind == "gaussian":
            key = ("gaussian_1d", str(self.device))
            if key not in self._kernels_torch:
                self._kernels_torch[key] = torch.from_numpy(self._gaussian_1d).to(self.device)
            one = self._kernels_torch[key]
            size = one.numel()
            # Full linear convolution: the minimal canvas is ROI + kernel - 1.
            padded = F.pad(isolated, (size-1, size-1, 0, 0))
            response = F.conv2d(padded, one[None, None, None, :])
            padded = F.pad(response, (0, 0, size-1, size-1))
            response = F.conv2d(padded, one[None, None, :, None])
        else:
            kernel = self.kernel(kind)
            kh, kw = kernel.shape[-2:]
            padded = F.pad(isolated, (kw-1, kw-1, kh-1, kh-1))
            response = F.conv2d(padded, kernel)
        peak = response.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (response * (255.0 / peak)).clamp(0, 255).to(torch.uint8)

    def prepare_batch(self, frame: np.ndarray, centers: Sequence[Tuple[float, float]],
                      kind: str):
        """Prepare every target together; returns results in target order."""
        extracted = [extract_padded_roi(frame, c, self.config.roi_size) for c in centers]
        rois = np.stack([x[0] for x in extracted])
        raw = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) for x in rois]
        if self.device.type == "cuda":
            isolated_t = self._isolate_gpu(rois)
            response_t = self._convolve_gpu(isolated_t, kind)
            isolated = isolated_t[:, 0].clamp(0, 255).to(torch.uint8).cpu().numpy()
            responses = response_t[:, 0].cpu().numpy()
        else:
            isolated = np.stack([self._isolate_cpu(x) for x in rois])
            responses = np.stack([self._convolve_cpu(x, kind) for x in isolated])
        pad = self.config.kernel_size // 2
        return [PreparedROI(raw[i], isolated[i], responses[i], extracted[i][1],
                            (extracted[i][2][0]-pad, extracted[i][2][1]-pad))
                for i in range(len(centers))]

    def prepare_dual_batch(self, frame: np.ndarray, centers: Sequence[Tuple[float, float]],
                           return_device: bool = False, gaussian_to_cpu: bool = True):
        """Reuse one crop/MVGC pass and optionally retain Gaussian responses on CUDA."""
        extracted = [extract_padded_roi(frame, c, self.config.roi_size) for c in centers]
        rois = np.stack([x[0] for x in extracted])
        raw = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) for x in rois]
        if self.device.type == "cuda":
            isolated_t = self._isolate_gpu(rois)
            circ_t = self._convolve_gpu(isolated_t, "circular")
            gauss_t = self._convolve_gpu(isolated_t, "gaussian")
            isolated = isolated_t[:, 0].clamp(0, 255).to(torch.uint8).cpu().numpy()
            circ = circ_t[:, 0].cpu().numpy()
            gauss_device = gauss_t[:, 0]
            gauss = gauss_device.cpu().numpy() if gaussian_to_cpu else None
        else:
            isolated = np.stack([self._isolate_cpu(x) for x in rois])
            circ = np.stack([self._convolve_cpu(x, "circular") for x in isolated])
            gauss = np.stack([self._convolve_cpu(x, "gaussian") for x in isolated])
            gauss_device = gauss
        def build(responses):
            return [PreparedROI(raw[i], isolated[i],
                                (responses[i] if responses is not None else np.empty((0, 0), np.uint8)),
                                extracted[i][1],
                                (extracted[i][2][0]-self.config.kernel_size//2,
                                 extracted[i][2][1]-self.config.kernel_size//2))
                    for i in range(len(centers))]
        result = (build(circ), build(gauss))
        return (*result, gauss_device) if return_device else result

    def apply(self, image: np.ndarray, kind: str) -> np.ndarray:
        """Compatibility helper for one already-isolated ROI."""
        if self.device.type == "cuda":
            x = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)
            return self._convolve_gpu(x, kind)[0, 0].cpu().numpy()
        return self._convolve_cpu(image, kind)

    def prepare(self, frame: np.ndarray, center: Tuple[float, float], kind: str):
        return self.prepare_batch(frame, [center], kind)[0]

"""Batched ROI-confined phase reconstruction and motion magnification."""

import cv2 as cv
import numpy as np
import torch


class BatchPhaseMagnifier:
    """Cache all base phases and amplify every target in one FFT batch."""
    def __init__(self, base_responses: np.ndarray, factor: float, device: torch.device,
                 denoise_kernel: int = 0, padding: int = 0):
        self.device = device
        self.factor = float(factor)
        self.denoise_kernel = denoise_kernel
        if isinstance(padding, (tuple, list)):
            self.pad_before, self.pad_after = map(int, padding)
        else:
            self.pad_before = self.pad_after = max(0, int(padding))
        if device.type == "cuda":
            if torch.is_tensor(base_responses):
                base_t = base_responses.to(device=device, dtype=torch.float32)
            else:
                base_t = torch.from_numpy(np.asarray(base_responses, dtype=np.float32)).to(device)
            if self.pad_before or self.pad_after:
                base_t = torch.nn.functional.pad(
                    base_t, (self.pad_before, self.pad_after,
                             self.pad_before, self.pad_after))
            self.base_phase = torch.angle(torch.fft.fft2(base_t))
        else:
            base = np.asarray(base_responses, dtype=np.float32)
            if self.pad_before or self.pad_after:
                base = np.pad(base, ((0, 0), (self.pad_before, self.pad_after),
                                     (self.pad_before, self.pad_after)))
            self.base_phase = np.angle(np.fft.fft2(base, axes=(-2, -1)))

    def amplify_batch(self, current_responses: np.ndarray) -> np.ndarray:
        if self.device.type == "cuda":
            if torch.is_tensor(current_responses):
                x = current_responses.to(device=self.device, dtype=torch.float32)
            else:
                x = torch.from_numpy(np.asarray(current_responses, dtype=np.float32)).to(self.device)
            if self.pad_before or self.pad_after:
                x = torch.nn.functional.pad(
                    x, (self.pad_before, self.pad_after,
                        self.pad_before, self.pad_after))
            spectrum = torch.fft.fft2(x)
            delta = torch.remainder(torch.angle(spectrum) - self.base_phase + torch.pi,
                                    2 * torch.pi) - torch.pi
            phase = self.base_phase + self.factor * delta
            rebuilt = torch.fft.ifft2(torch.abs(spectrum) * torch.exp(1j * phase)).real.abs()
            peak = rebuilt.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            output = (rebuilt * (255.0 / peak)).clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            current = np.asarray(current_responses, dtype=np.float32)
            if self.pad_before or self.pad_after:
                current = np.pad(current, ((0, 0), (self.pad_before, self.pad_after),
                                           (self.pad_before, self.pad_after)))
            spectrum = np.fft.fft2(current, axes=(-2, -1))
            delta = (np.angle(spectrum) - self.base_phase + np.pi) % (2*np.pi) - np.pi
            phase = self.base_phase + self.factor * delta
            rebuilt = np.abs(np.fft.ifft2(np.abs(spectrum) * np.exp(1j*phase), axes=(-2, -1)).real)
            peak = np.maximum(rebuilt.max(axis=(-2, -1), keepdims=True), 1.0)
            output = np.clip(rebuilt * (255.0 / peak), 0, 255).astype(np.uint8)
        if self.denoise_kernel:
            output = np.stack([cv.GaussianBlur(x, (self.denoise_kernel, self.denoise_kernel), 0)
                               for x in output])
        return output


class PhaseMagnifier(BatchPhaseMagnifier):
    """Backward-compatible single-target wrapper."""
    def __init__(self, base_response, factor, device, denoise_kernel=0, padding=0):
        super().__init__(np.asarray(base_response)[None], factor, device,
                         denoise_kernel, padding)

    def amplify(self, current_response):
        return self.amplify_batch(np.asarray(current_response)[None])[0]


class BatchPhaseShiftEstimator:
    """Estimate fixed-ROI translations without reconstructing magnified images."""
    def __init__(self, base_responses):
        if torch.is_tensor(base_responses):
            base_responses = base_responses.detach().cpu().numpy()
        self.base = np.asarray(base_responses, dtype=np.float32)
        h, w = self.base.shape[-2:]
        self.window = cv.createHanningWindow((w, h), cv.CV_32F)

    def estimate_batch(self, current_responses):
        if torch.is_tensor(current_responses):
            current_responses = current_responses.detach().cpu().numpy()
        current = np.asarray(current_responses, dtype=np.float32)
        shifts, confidence = [], []
        for base, frame in zip(self.base, current):
            shift, response = cv.phaseCorrelate(base, frame, self.window)
            shifts.append((float(shift[0]), float(shift[1])))
            confidence.append(float(response))
        return np.asarray(shifts, dtype=np.float32), np.asarray(confidence, dtype=np.float32)

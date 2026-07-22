"""Central configuration for both ConBo demos."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ConBoConfig:
    input_video: Path
    output_dir: Path = Path("outputs")
    run_name: str = "conbo"
    base_frame: int = 0
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    roi_size: int = 60
    gamma: float = 5.0
    threshold: int = 120
    adaptive_mvgc: bool = True
    adaptive_samples_per_second: int = 20
    adaptive_sample_seconds: float = 1.0
    adaptive_sample_start_frame: int = 0
    adaptive_random_seed: int = 2026
    adaptive_min_circularity: float = 0.85
    adaptive_full_validation_candidates: int = 5
    adaptive_exact_validation_candidates: int = 2
    adaptive_search_mode: str = "fast"
    morphology_size: int = 3
    morphology_shape: str = "ellipse"
    kernel_size: int = 60
    circular_radius_ratio: float = 0.5
    gaussian_sigma: float = 10.0
    amplification_factor: float = 15.0
    mag_algorithm: str = "phase_reconstruct"
    phase_min_confidence: float = 0.20
    phase_max_shift_px: float = 10.0
    mag_reconstruction_padding: Optional[int] = None
    update_strategy: str = "detected"
    device: str = "auto"
    gpu_min_batch_pixels: int = 200000
    min_response: float = 0.03
    min_circularity: float = 0.70
    mag_denoise_kernel: int = 0
    output_fps: Optional[float] = None
    codec: str = "mp4v"
    io_acceleration: str = "cpu"
    write_video: bool = True
    output_scale: float = 1.0
    show_steps: bool = True
    show_live: bool = False
    draw_coordinates: bool = True
    draw_trajectory: bool = True
    trajectory_seconds: float = 1.0
    display_max_width: int = 1600
    display_max_height: int = 900
    console_interval: int = 1

    def validate(self) -> None:
        """Reject ambiguous or unsafe parameter combinations early."""
        self.input_video = Path(self.input_video)
        self.output_dir = Path(self.output_dir)
        if not self.input_video.exists():
            raise FileNotFoundError(self.input_video)
        if self.roi_size < 8 or self.kernel_size < 3:
            raise ValueError("roi_size must be >= 8 and kernel_size >= 3")
        if self.gamma <= 0 or not 0 <= self.threshold <= 255:
            raise ValueError("gamma must be positive and threshold must be in [0, 255]")
        if self.morphology_size < 1 or self.morphology_size % 2 == 0:
            raise ValueError("morphology_size must be a positive odd integer")
        if self.morphology_shape not in {"rect", "ellipse", "cross"}:
            raise ValueError("morphology_shape must be 'rect', 'ellipse', or 'cross'")
        if self.adaptive_samples_per_second < 1:
            raise ValueError("adaptive_samples_per_second must be at least 1")
        if self.adaptive_sample_seconds <= 0:
            raise ValueError("adaptive_sample_seconds must be positive")
        if not 0.0 < self.adaptive_min_circularity <= 1.0:
            raise ValueError("adaptive_min_circularity must be in (0, 1]")
        if self.adaptive_full_validation_candidates < 1:
            raise ValueError("adaptive_full_validation_candidates must be at least 1")
        if self.adaptive_exact_validation_candidates < 1:
            raise ValueError("adaptive_exact_validation_candidates must be at least 1")
        if self.adaptive_search_mode not in {"fast", "exhaustive"}:
            raise ValueError("adaptive_search_mode must be 'fast' or 'exhaustive'")
        if self.update_strategy not in {"detected", "fixed"}:
            raise ValueError("update_strategy must be 'detected' or 'fixed'")
        if self.mag_algorithm not in {"phase_reconstruct", "phase_shift"}:
            raise ValueError("mag_algorithm must be 'phase_reconstruct' or 'phase_shift'")
        if not 0.0 <= self.phase_min_confidence <= 1.0:
            raise ValueError("phase_min_confidence must be in [0, 1]")
        if self.phase_max_shift_px <= 0:
            raise ValueError("phase_max_shift_px must be positive")
        if self.mag_reconstruction_padding is not None and self.mag_reconstruction_padding < 0:
            raise ValueError("mag_reconstruction_padding must be None or >= 0")
        if len(self.codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if self.io_acceleration not in {"auto", "cpu", "hardware"}:
            raise ValueError("io_acceleration must be 'auto', 'cpu', or 'hardware'")
        if not 0.1 <= self.output_scale <= 1.0:
            raise ValueError("output_scale must be in [0.1, 1.0]")
        if self.mag_denoise_kernel not in {0, 3, 5, 7}:
            raise ValueError("mag_denoise_kernel must be 0, 3, 5, or 7")
        if not 0.0 < self.min_circularity <= 1.0:
            raise ValueError("min_circularity must be in (0, 1]")
        if self.console_interval < 0:
            raise ValueError("console_interval must be >= 0")
        if not self.run_name or any(c in self.run_name for c in '<>:"/\\|?*'):
            raise ValueError("run_name must be a non-empty filename-safe name")
        self.output_dir.mkdir(parents=True, exist_ok=True)

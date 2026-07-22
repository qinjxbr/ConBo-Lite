"""One optimized 12-step ConBo pipeline with Track and fixed-ROI MAG modes."""

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence, Tuple
import cv2 as cv
import numpy as np

from .config import ConBoConfig
from .detection import Detection, EllipseDetector
from .magnification import BatchPhaseMagnifier, BatchPhaseShiftEstimator
from .preprocessing import Convolver, PreparedROI
from .video_io import (draw_detection, draw_history, fit_for_display, read_frame,
                       write_csv, open_video_capture, create_video_writer)


@dataclass
class TargetState:
    target_id: int
    selected: Tuple[float, float]
    center: Tuple[float, float]
    base_center: Tuple[float, float]
    detection: Detection
    missed: int = 0
    history: deque = field(default_factory=deque)


def phase_shift_is_valid(shift, confidence: float, min_confidence: float,
                         max_shift_px: float) -> bool:
    """Reject empty-response and implausibly large phase-correlation results."""
    values = np.asarray(shift, dtype=float)
    return bool(values.shape == (2,) and np.all(np.isfinite(values)) and
                np.isfinite(confidence) and confidence >= min_confidence and
                np.linalg.norm(values) <= max_shift_px)


class ConBoPipeline:
    """Tracking is the base; MAG shares it but fixes spatial sampling for phase."""
    def __init__(self, config: ConBoConfig):
        config.validate()
        self.cfg = config
        self.convolver = Convolver(config)
        self.detector = EllipseDetector()

    def _global_detection(self, prepared: PreparedROI, fallback_center) -> Detection:
        local = self.detector.detect(prepared.response)
        if local is None:
            return Detection(tuple(map(float, fallback_center)), (8.0, 8.0), 0.0, 0.0, "hold")
        return Detection((prepared.anchor[0] + local.center[0],
                          prepared.anchor[1] + local.center[1]),
                         local.axes, local.angle, local.score, local.method)

    def _valid(self, detection: Detection):
        return (detection.method != "hold" and
                detection.score >= self.cfg.min_response and
                detection.circularity >= self.cfg.min_circularity)

    def initialize(self, points: Sequence[Tuple[int, int]], mode: str):
        self.convolver.configure(len(points), mode)
        base = read_frame(self.cfg.input_video, self.cfg.base_frame,
                          self.cfg.io_acceleration)
        if base is None:
            raise RuntimeError(f"Cannot read base frame {self.cfg.base_frame}")
        selected = [tuple(map(float, p)) for p in points]
        if mode == "mag":
            base_track, base_gauss, base_gauss_batch = self.convolver.prepare_dual_batch(
                base, selected, return_device=True, gaussian_to_cpu=True)
            detections = [self._global_detection(x, p) for x, p in zip(base_track, selected)]
        else:
            first = self.convolver.prepare_batch(base, selected, "circular")
            first_detections = [self._global_detection(x, p) for x, p in zip(first, selected)]
            recentered = [d.center for d in first_detections]
            base_track = self.convolver.prepare_batch(base, recentered, "circular")
            detections = [self._global_detection(x, p) for x, p in zip(base_track, recentered)]
            base_gauss = []
            base_gauss_batch = None
        states = [TargetState(i, selected[i-1], d.center, d.center, d)
                  for i, d in enumerate(detections, 1)]
        return base, states, base_track, base_gauss, base_gauss_batch

    def run(self, points: Sequence[Tuple[int, int]], mode: str = "track") -> dict:
        """Execute the shared method and write annotated MP4 plus tidy CSV."""
        if mode not in {"track", "mag"} or not points:
            raise ValueError("mode must be track/mag and at least one point is required")
        _, states, _, base_gauss, base_gauss_batch = self.initialize(points, mode)
        magnifier = None
        shift_estimator = None
        response_size = self.cfg.roi_size + self.cfg.kernel_size - 1
        if self.cfg.mag_reconstruction_padding is None:
            minimum_size = response_size + 2*(self.cfg.kernel_size//2)
            fft_size = cv.getOptimalDFTSize(minimum_size)
            extra = fft_size-response_size
            reconstruction_padding = (extra//2, extra-extra//2)
        else:
            reconstruction_padding = (self.cfg.mag_reconstruction_padding,
                                      self.cfg.mag_reconstruction_padding)
        base_mag_centers = []
        if mode == "mag":
            if self.cfg.mag_algorithm == "phase_reconstruct":
                magnifier = BatchPhaseMagnifier(
                    base_gauss_batch,
                    self.cfg.amplification_factor, self.convolver.device,
                    self.cfg.mag_denoise_kernel, reconstruction_padding)
            else:
                shift_estimator = BatchPhaseShiftEstimator(base_gauss_batch)
            for state, prepared in zip(states, base_gauss):
                local = self.detector.detect(prepared.response)
                base_mag_centers.append(
                    state.base_center if local is None else
                    (prepared.anchor[0]+local.center[0], prepared.anchor[1]+local.center[1]))

        cap, decode_backend = open_video_capture(self.cfg.input_video,
                                                 self.cfg.io_acceleration)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.cfg.input_video}")
        first_frame = self.cfg.start_frame or 0
        if first_frame > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, first_frame)
        fps_in = float(cap.get(cv.CAP_PROP_FPS))
        fps = self.cfg.output_fps or (fps_in if np.isfinite(fps_in) and fps_in > 0 else 30.0)
        width, height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_path = (self.cfg.output_dir / f"{self.cfg.run_name}_{mode}.mp4"
                      if self.cfg.write_video else None)
        writer = None
        encode_backend = "disabled"
        if video_path is not None:
            output_size = (max(1, int(round(width*self.cfg.output_scale))),
                           max(1, int(round(height*self.cfg.output_scale))))
            writer, encode_backend = create_video_writer(
                video_path, self.cfg.codec, fps, output_size, self.cfg.io_acceleration)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create {video_path}")
        history_length = max(2, int(round(fps * self.cfg.trajectory_seconds)))
        for state in states:
            state.history = deque(maxlen=history_length)

        rows, frame_index, processed = [], first_frame-1, 0
        timing = {"preprocess_s": 0.0, "detect_s": 0.0, "phase_s": 0.0,
                  "draw_write_s": 0.0}
        run_start = perf_counter()
        live_name = f"ConBo {mode.upper()} - Esc to stop"
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_index += 1
            if self.cfg.start_frame is not None and frame_index < self.cfg.start_frame: continue
            if self.cfg.end_frame is not None and frame_index > self.cfg.end_frame: break
            roi_centers = ([s.selected for s in states] if mode == "mag" else
                           [s.center if self.cfg.update_strategy == "detected" else s.selected
                            for s in states])
            t = perf_counter()
            if mode == "mag":
                prepared_all, gaussian_all, gaussian_batch = self.convolver.prepare_dual_batch(
                    frame, roi_centers, return_device=True, gaussian_to_cpu=False)
            else:
                prepared_all = self.convolver.prepare_batch(frame, roi_centers, "circular")
                gaussian_all = []
                gaussian_batch = None
            timing["preprocess_s"] += perf_counter()-t

            amplified_all = phase_shifts = phase_confidence = None
            if mode == "mag":
                t = perf_counter()
                if self.cfg.mag_algorithm == "phase_reconstruct":
                    amplified_all = magnifier.amplify_batch(gaussian_batch)
                else:
                    phase_shifts, phase_confidence = shift_estimator.estimate_batch(gaussian_batch)
                if self.convolver.device.type == "cuda":
                    import torch
                    torch.cuda.synchronize()
                timing["phase_s"] += perf_counter()-t

            # Track always performs the full circular response + EdgeDrawing path on
            # every target in every frame. No centroid/hybrid shortcut is permitted.
            output = frame.copy()
            for j, (state, prepared, roi_center) in enumerate(
                    zip(states, prepared_all, roi_centers)):
                t = perf_counter()
                raw_detection = self._global_detection(prepared, state.center)
                track_valid = self._valid(raw_detection)
                if track_valid:
                    state.center, state.detection, state.missed = raw_detection.center, raw_detection, 0
                else:
                    state.missed += 1
                original_center = state.center
                displayed = state.detection
                mag_valid = track_valid
                amp_dx = amp_dy = 0.0
                phase_dx = phase_dy = phase_score = 0.0
                if mode == "mag" and self.cfg.mag_algorithm == "phase_reconstruct":
                    amp_local = self.detector.detect(amplified_all[j])
                    mag_valid = bool(track_valid and amp_local is not None and
                                     amp_local.circularity >= self.cfg.min_circularity)
                    if mag_valid:
                        gaussian = gaussian_all[j]
                        amp_center = (gaussian.anchor[0]-reconstruction_padding[0]+amp_local.center[0],
                                      gaussian.anchor[1]-reconstruction_padding[0]+amp_local.center[1])
                        amp_dx = amp_center[0] - base_mag_centers[j][0]
                        amp_dy = amp_center[1] - base_mag_centers[j][1]
                        displayed = Detection(amp_center, amp_local.axes, amp_local.angle,
                                              amp_local.score, amp_local.method)
                    else:
                        displayed = Detection(original_center, state.detection.axes,
                                              state.detection.angle, 0.0,
                                              "hold_mag_invalid")
                elif mode == "mag":
                    shift_x, shift_y = map(float, phase_shifts[j])
                    phase_dx, phase_dy = shift_x, shift_y
                    phase_score = float(phase_confidence[j])
                    mag_valid = bool(track_valid and phase_shift_is_valid(
                        (shift_x, shift_y), phase_score,
                        self.cfg.phase_min_confidence, self.cfg.phase_max_shift_px))
                    # The current circle remains detector-derived. Phase correlation adds
                    # only the extra (alpha-1) displacement needed for magnification.
                    if mag_valid:
                        extra = self.cfg.amplification_factor - 1.0
                        amp_center = (original_center[0] + extra*shift_x,
                                      original_center[1] + extra*shift_y)
                        amp_dx = amp_center[0] - state.base_center[0]
                        amp_dy = amp_center[1] - state.base_center[1]
                        displayed = Detection(amp_center, state.detection.axes,
                                              state.detection.angle, phase_score,
                                              f"{state.detection.method}+phase_shift")
                    else:
                        displayed = Detection(original_center, state.detection.axes,
                                              state.detection.angle, 0.0,
                                              "hold_phase_invalid")
                timing["detect_s"] += perf_counter()-t
                dx, dy = original_center[0]-state.base_center[0], original_center[1]-state.base_center[1]
                state.history.append(original_center)
                if self.cfg.draw_trajectory:
                    draw_history(output, state.history)
                draw_detection(output, state.target_id, displayed.center, displayed.axes,
                               displayed.angle, state.base_center,
                               mag_valid if mode == "mag" else track_valid,
                               self.cfg.draw_coordinates)
                rows.append({"frame": frame_index, "target_id": state.target_id,
                             "roi_center_x": f"{roi_center[0]:.6f}",
                             "roi_center_y": f"{roi_center[1]:.6f}",
                             "x": f"{original_center[0]:.6f}", "y": f"{original_center[1]:.6f}",
                             "dx": f"{dx:.6f}", "dy": f"{dy:.6f}",
                             "amplified_dx": f"{amp_dx:.6f}", "amplified_dy": f"{amp_dy:.6f}",
                             "phase_dx": f"{phase_dx:.6f}", "phase_dy": f"{phase_dy:.6f}",
                             "phase_confidence": f"{phase_score:.6f}",
                             "display_x": f"{displayed.center[0]:.6f}",
                             "display_y": f"{displayed.center[1]:.6f}",
                             "radius": f"{displayed.radius:.6f}",
                             "circularity": f"{displayed.circularity:.6f}",
                             "valid": int(track_valid),
                             "track_valid": int(track_valid),
                             "mag_valid": int(mag_valid),
                             "detector": displayed.method,
                             "mag_algorithm": self.cfg.mag_algorithm if mode == "mag" else "none",
                             "reconstruction_padding": (f"{reconstruction_padding[0]}/"
                                                        f"{reconstruction_padding[1]}" if
                                                        mode == "mag" and self.cfg.mag_algorithm == "phase_reconstruct"
                                                        else 0),
                             "roi_policy": "fixed_initial" if mode == "mag" else self.cfg.update_strategy,
                             "detection_policy": "full_circular_edgedrawing_every_frame"})
                if (mode == "mag" and self.cfg.console_interval and
                        processed % self.cfg.console_interval == 0):
                    print(f"frame={frame_index} id={state.target_id} "
                          f"track_valid={int(track_valid)} mag_valid={int(mag_valid)} "
                          f"original=({original_center[0]:.3f},{original_center[1]:.3f}) "
                          f"amp_disp=({amp_dx:.3f},{amp_dy:.3f}) "
                          f"amplified=({displayed.center[0]:.3f},{displayed.center[1]:.3f})")
            t = perf_counter()
            if writer is not None:
                written = (output if self.cfg.output_scale == 1.0 else
                           cv.resize(output, output_size, interpolation=cv.INTER_AREA))
                writer.write(written)
            processed += 1
            if self.cfg.show_live:
                shown, _ = fit_for_display(output, self.cfg.display_max_width,
                                           self.cfg.display_max_height)
                cv.imshow(live_name, shown)
                if cv.waitKey(1) & 0xFF == 27: break
            timing["draw_write_s"] += perf_counter()-t
        elapsed = perf_counter()-run_start
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        csv_path = self.cfg.output_dir / f"{self.cfg.run_name}_{mode}.csv"
        write_csv(csv_path, rows)
        expected = processed * len(states)
        if len(rows) != expected:
            raise AssertionError(f"CSV rows {len(rows)} != {expected}")
        return {"video": video_path, "csv": csv_path, "frames": processed,
                "targets": len(states), "device": str(self.convolver.device),
                "mag_algorithm": self.cfg.mag_algorithm if mode == "mag" else "none",
                "reconstruction_padding": (f"{reconstruction_padding[0]}/"
                                           f"{reconstruction_padding[1]}" if
                                           mode == "mag" and self.cfg.mag_algorithm == "phase_reconstruct"
                                           else 0),
                "roi_policy": "fixed_initial" if mode == "mag" else self.cfg.update_strategy,
                "detection_policy": "full_circular_edgedrawing_every_frame",
                "decode_backend": decode_backend, "encode_backend": encode_backend,
                "output_scale": self.cfg.output_scale,
                "elapsed_s": elapsed, "processing_fps": processed/max(elapsed, 1e-9),
                "timing": timing}

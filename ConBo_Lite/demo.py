"""Editable and command-line entry for the unified ConBo Lite method."""

import argparse
from pathlib import Path
import cv2 as cv
import numpy as np

from conbo import ConBoConfig, ConBoPipeline
from conbo.adaptive_mvgc import tune_adaptive_mvgc
from conbo.video_io import (draw_detection, overlay_response, pick_points,
                            read_frame, show_image)


# =============================================================================
# EDIT THESE SETTINGS, THEN CLICK "RUN" IN YOUR IDE
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent

RUN_NAME = "my_conbo_run"                 # Prefix of output MP4 and CSV
MODE = "track"                              # "track" or "mag"
INPUT_VIDEO = PROJECT_DIR / "input" / "input1.MOV"
OUTPUT_DIR = PROJECT_DIR / "outputs"

BASE_FRAME = 0                            # Zero-based reference frame
START_FRAME = None                        # None means the first frame
END_FRAME = None                          # None means the last frame
POINTS = None                             # None: mouse click; or [(x1,y1), (x2,y2)]

ROI_SIZE = 60                             # Fixed MAG window / tracking search ROI
MVGC_GAMMA = 5.0
MVGC_THRESHOLD = 100
ADAPTIVE_MVGC = True                      # Search around the two values above before full processing
ADAPTIVE_DURATION_SECONDS = 2             # 2 -> 40 frames; 1 -> 20; 0.1 -> 2
ADAPTIVE_SAMPLES_PER_SECOND = 20           # Random frames per second (partial seconds scale proportionally)
ADAPTIVE_SAMPLE_START_FRAME = 0
ADAPTIVE_RANDOM_SEED = 2026                # Reproducible random sampling
ADAPTIVE_MIN_CIRCULARITY = 0.85
ADAPTIVE_FULL_VALIDATION_CANDIDATES = 5    # Recheck the best candidates on every calibration frame
ADAPTIVE_EXACT_VALIDATION_CANDIDATES = 2   # Full convolution/EdgeDrawing after cheap mask screening
ADAPTIVE_SEARCH_MODE = "exhaustive"        # "fast": proxy pre-screen + exact finalists; "exhaustive": exact all
MORPHOLOGY_SIZE = 3                       # Positive odd number
MORPHOLOGY_SHAPE = "ellipse"              # "ellipse" is robust for tiny spots; also "rect"/"cross"
KERNEL_SIZE = 100
CIRCULAR_RADIUS_RATIO = 0.5
GAUSSIAN_SIGMA = 10.0
AMPLIFICATION_FACTOR = 50.0               # Used only by MAG
MAG_ALGORITHM = "phase_reconstruct"       # Default exact IFFT mode; "phase_shift" is fast
PHASE_MIN_CONFIDENCE = 0.20               # Fast MAG: reject unreliable phase correlation
PHASE_MAX_SHIFT_PX = 10.0                 # Fast MAG: reject implausible raw shifts before amplification
MAG_RECONSTRUCTION_PADDING = None         # Exact MAG: None adds >=KERNEL_SIZE//2 and picks a fast FFT size

TRACK_UPDATE_STRATEGY = "detected"        # Track only: "detected" or "fixed"
DEVICE = "auto"                          # "auto", "cpu", or "cuda"
GPU_MIN_BATCH_PIXELS = 200000             # Auto: GPU for sufficiently large target batches
MAG_DENOISE_KERNEL = 0                    # 0 is fastest; optional 3/5/7 Gaussian cleanup
OUTPUT_FPS = None                         # None keeps source FPS
CODEC = "mp4v"
IO_ACCELERATION = "cpu"                   # "cpu" is quiet/reliable; "auto"/"hardware" tries OpenCV HW I/O
WRITE_VIDEO = False                        # False: fastest CSV-only analysis
OUTPUT_SCALE = 1.0                        # Video only: 1.0 native, 0.5 quarter pixel count
SHOW_STEPS = True
SHOW_LIVE = True
DRAW_COORDINATES = True
DRAW_TRAJECTORY = True
TRAJECTORY_SECONDS = 1.0
DISPLAY_MAX_WIDTH = 1600                  # Display only; never changes analysis pixels
DISPLAY_MAX_HEIGHT = 900
CONSOLE_INTERVAL = 1                      # MAG: 1=every frame, 0=off, 30=every 30 frames
# =============================================================================


def parse_args():
    """CLI values override the editable settings; no arguments uses them directly."""
    p = argparse.ArgumentParser(description="ConBo Lite tracking and motion magnification")
    p.add_argument("--name", default=RUN_NAME)
    p.add_argument("--mode", choices=("track", "mag"), default=MODE)
    p.add_argument("--input", type=Path, default=INPUT_VIDEO)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--base-frame", type=int, default=BASE_FRAME)
    p.add_argument("--start-frame", type=int, default=START_FRAME)
    p.add_argument("--end-frame", type=int, default=END_FRAME)
    p.add_argument("--points", nargs="*", help="x,y pairs; omit for mouse selection")
    p.add_argument("--roi-size", type=int, default=ROI_SIZE)
    p.add_argument("--gamma", type=float, default=MVGC_GAMMA)
    p.add_argument("--threshold", type=int, default=MVGC_THRESHOLD)
    p.add_argument("--adaptive-mvgc", action=argparse.BooleanOptionalAction,
                   default=ADAPTIVE_MVGC)
    p.add_argument("--adaptive-samples-per-second", "--adaptive-samples", type=int,
                   dest="adaptive_samples_per_second",
                   default=ADAPTIVE_SAMPLES_PER_SECOND)
    p.add_argument("--adaptive-seconds", type=float, default=ADAPTIVE_DURATION_SECONDS)
    p.add_argument("--adaptive-start-frame", type=int, default=ADAPTIVE_SAMPLE_START_FRAME)
    p.add_argument("--adaptive-seed", type=int, default=ADAPTIVE_RANDOM_SEED)
    p.add_argument("--adaptive-full-validation-candidates", type=int,
                   default=ADAPTIVE_FULL_VALIDATION_CANDIDATES)
    p.add_argument("--adaptive-exact-validation-candidates", type=int,
                   default=ADAPTIVE_EXACT_VALIDATION_CANDIDATES)
    p.add_argument("--adaptive-search-mode", choices=("fast", "exhaustive"),
                   default=ADAPTIVE_SEARCH_MODE)
    p.add_argument("--morphology-size", type=int, default=MORPHOLOGY_SIZE)
    p.add_argument("--morphology-shape", choices=("rect", "ellipse", "cross"),
                   default=MORPHOLOGY_SHAPE)
    p.add_argument("--kernel-size", type=int, default=KERNEL_SIZE)
    p.add_argument("--circular-radius-ratio", type=float, default=CIRCULAR_RADIUS_RATIO)
    p.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    p.add_argument("--amplification", type=float, default=AMPLIFICATION_FACTOR)
    p.add_argument("--mag-algorithm", choices=("phase_reconstruct", "phase_shift"),
                   default=MAG_ALGORITHM)
    p.add_argument("--phase-min-confidence", type=float, default=PHASE_MIN_CONFIDENCE)
    p.add_argument("--phase-max-shift-px", type=float, default=PHASE_MAX_SHIFT_PX)
    p.add_argument("--mag-reconstruction-padding", type=int,
                   default=MAG_RECONSTRUCTION_PADDING)
    p.add_argument("--track-update", choices=("detected", "fixed"), default=TRACK_UPDATE_STRATEGY)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    p.add_argument("--gpu-min-batch-pixels", type=int, default=GPU_MIN_BATCH_PIXELS)
    p.add_argument("--mag-denoise-kernel", type=int, choices=(0, 3, 5, 7),
                   default=MAG_DENOISE_KERNEL)
    p.add_argument("--output-fps", type=float, default=OUTPUT_FPS)
    p.add_argument("--codec", default=CODEC)
    p.add_argument("--io-acceleration", choices=("auto", "cpu", "hardware"),
                   default=IO_ACCELERATION)
    p.add_argument("--write-video", action=argparse.BooleanOptionalAction,
                   default=WRITE_VIDEO)
    p.add_argument("--output-scale", type=float, default=OUTPUT_SCALE)
    p.add_argument("--show-live", action=argparse.BooleanOptionalAction, default=SHOW_LIVE)
    p.add_argument("--show-steps", action=argparse.BooleanOptionalAction, default=SHOW_STEPS)
    p.add_argument("--draw-coordinates", action=argparse.BooleanOptionalAction,
                   default=DRAW_COORDINATES)
    p.add_argument("--draw-trajectory", action=argparse.BooleanOptionalAction,
                   default=DRAW_TRAJECTORY)
    p.add_argument("--trajectory-seconds", type=float, default=TRAJECTORY_SECONDS)
    p.add_argument("--console-interval", type=int, default=CONSOLE_INTERVAL)
    return p.parse_args()


def _tile(image, title, subtitle="", highlight=False, tile_size=280):
    """Build one clearly labeled tutorial tile."""
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image = cv.resize(image, (tile_size, tile_size),
                      interpolation=cv.INTER_NEAREST)
    canvas = cv.copyMakeBorder(image, 62, 12, 12, 12, cv.BORDER_CONSTANT,
                               value=(35, 35, 35))
    color = (0, 255, 0) if highlight else (255, 255, 255)
    cv.putText(canvas, title, (12, 24), cv.FONT_HERSHEY_SIMPLEX, .58,
               color, 1, cv.LINE_AA)
    cv.putText(canvas, subtitle, (12, 48), cv.FONT_HERSHEY_SIMPLEX, .42,
               (190, 190, 190), 1, cv.LINE_AA)
    if highlight:
        cv.rectangle(canvas, (1, 1), (canvas.shape[1]-2, canvas.shape[0]-2),
                     (0, 255, 0), 3)
    return canvas


def _arrow(height):
    image = np.full((height, 54, 3), 35, np.uint8)
    cv.arrowedLine(image, (6, height//2), (47, height//2), (0, 220, 255),
                   2, cv.LINE_AA, tipLength=.28)
    return image


def show_method_steps(pipeline, frame, points, mode, config):
    """Show the old Lite teaching sequence as one labeled, non-ambiguous flow."""
    print("Tutorial: each target is shown as Raw ROI -> MVGC isolation -> FINAL circular response.")
    print("The green-bordered response is the response actually used for circle detection.")
    for target_id, point in enumerate(points, 1):
        circular = pipeline.convolver.prepare(frame, point, "circular")
        found = pipeline.detector.detect(circular.response)
        final = cv.cvtColor(circular.response, cv.COLOR_GRAY2BGR)
        circularity = 0.0
        if found is not None:
            circularity = found.circularity
            center = tuple(map(lambda x: int(round(x)), found.center))
            cv.circle(final, center, max(2, int(round(found.radius))), (0, 255, 0), 1, cv.LINE_AA)
        tiles = [
            _tile(circular.raw_gray, "1  Raw local ROI", "Input around selected target"),
            _tile(circular.isolated, "2  MVGC + morphology", "Background suppressed"),
            _tile(final, "3  FINAL circular Bokeh", f"USED FOR DETECTION  circ={circularity:.3f}", True),
        ]
        panel = cv.hconcat([tiles[0], _arrow(tiles[0].shape[0]), tiles[1],
                            _arrow(tiles[0].shape[0]), tiles[2]])
        if not show_image(f"Step 2/3 - ID{target_id} processing flow", panel,
                          config.display_max_width, config.display_max_height):
            return
        overlay = overlay_response(frame, circular.response, circular.anchor)
        if found is not None:
            global_center = (circular.anchor[0]+found.center[0],
                             circular.anchor[1]+found.center[1])
            draw_detection(overlay, target_id, global_center, found.axes, found.angle,
                           global_center, True, True)
        if not show_image(f"Step 3 - ID{target_id} final response on base frame", overlay,
                          config.display_max_width, config.display_max_height):
            return
        if mode == "mag":
            gaussian = pipeline.convolver.prepare(frame, point, "gaussian")
            phase_use = ("Fixed ROI; IFFT phase reconstruction" if
                         config.mag_algorithm == "phase_reconstruct" else
                         "Fixed ROI; fast phase-shift estimation")
            gaussian_panel = _tile(gaussian.response, "MAG Gaussian response",
                                   phase_use, True, 360)
            if not show_image(f"Step 3b - ID{target_id} MAG phase response", gaussian_panel,
                              config.display_max_width, config.display_max_height):
                return


def main():
    a = parse_args()
    configured_points = POINTS if a.points is None else [tuple(map(int, x.split(","))) for x in a.points]
    config = ConBoConfig(
        input_video=a.input, output_dir=a.output_dir, run_name=a.name,
        base_frame=a.base_frame, start_frame=a.start_frame, end_frame=a.end_frame,
        roi_size=a.roi_size, gamma=a.gamma, threshold=a.threshold,
        adaptive_mvgc=a.adaptive_mvgc,
        adaptive_samples_per_second=a.adaptive_samples_per_second,
        adaptive_sample_seconds=a.adaptive_seconds,
        adaptive_sample_start_frame=a.adaptive_start_frame,
        adaptive_random_seed=a.adaptive_seed,
        adaptive_min_circularity=ADAPTIVE_MIN_CIRCULARITY,
        adaptive_full_validation_candidates=a.adaptive_full_validation_candidates,
        adaptive_exact_validation_candidates=a.adaptive_exact_validation_candidates,
        adaptive_search_mode=a.adaptive_search_mode,
        morphology_size=a.morphology_size, morphology_shape=a.morphology_shape,
        kernel_size=a.kernel_size,
        circular_radius_ratio=a.circular_radius_ratio, gaussian_sigma=a.gaussian_sigma,
        amplification_factor=a.amplification, mag_algorithm=a.mag_algorithm,
        phase_min_confidence=a.phase_min_confidence,
        phase_max_shift_px=a.phase_max_shift_px,
        mag_reconstruction_padding=a.mag_reconstruction_padding,
        update_strategy=a.track_update,
        device=a.device, gpu_min_batch_pixels=a.gpu_min_batch_pixels,
        mag_denoise_kernel=a.mag_denoise_kernel,
        output_fps=a.output_fps, codec=a.codec,
        io_acceleration=a.io_acceleration, write_video=a.write_video,
        output_scale=a.output_scale,
        show_steps=a.show_steps, show_live=a.show_live,
        draw_coordinates=a.draw_coordinates, draw_trajectory=a.draw_trajectory,
        trajectory_seconds=a.trajectory_seconds,
        display_max_width=DISPLAY_MAX_WIDTH, display_max_height=DISPLAY_MAX_HEIGHT,
        console_interval=a.console_interval)
    config.validate()
    frame = read_frame(a.input, a.base_frame, config.io_acceleration)
    if frame is None:
        raise RuntimeError(f"Base frame {a.base_frame} cannot be read from {a.input}")
    points = configured_points if configured_points is not None else pick_points(
        frame, config.display_max_width, config.display_max_height)
    if not points:
        print("No targets selected; nothing was written.")
        return
    if config.adaptive_mvgc:
        print("\nAdaptive MVGC: sampling a short video segment and testing local parameter combinations...")
        adaptive, report_path = tune_adaptive_mvgc(config, points)
        best = adaptive.top_candidates[0]
        print(
            f"Adaptive MVGC selected gamma={adaptive.gamma:.3f}, "
            f"threshold={adaptive.threshold} from initial "
            f"gamma={adaptive.initial_gamma:.3f}, threshold={adaptive.initial_threshold}."
        )
        print(
            f"Adaptive reference frame={adaptive.reference_frame} "
            f"(initial={adaptive.initial_reference_frame}), "
            f"mean displacement={adaptive.reference_mean_displacement_px:.3f}px, "
            f"max displacement={adaptive.reference_max_displacement_px:.3f}px."
        )
        print(
            f"Calibration: success={best.success_rate:.1%}, "
            f"position std={best.position_std_px:.3f}px, "
            f"position span={best.position_span_px:.3f}px, "
            f"center alignment={best.center_alignment_px:.3f}px "
            f"(stability={best.center_alignment_std_px:.3f}px), "
            f"radius CV={best.radius_cv:.3%}, "
            f"circularity={best.mean_circularity:.3f}, "
            f"search combinations={adaptive.candidates_tested}, "
            f"search={adaptive.search_mode}/{adaptive.proxy_backend} "
            f"(proxy={adaptive.proxy_candidates_tested}, exact-sample={adaptive.exact_sample_candidates}), "
            f"full mask screening={adaptive.full_validation_candidates}, "
            f"exact full validation={adaptive.exact_validation_candidates} candidates x "
            f"{adaptive.calibration_frames} frames, elapsed={adaptive.elapsed_s:.2f}s."
        )
        initial = adaptive.initial_metrics
        print(
            f"Initial-setting comparison: success={initial.success_rate:.1%}, "
            f"position std={initial.position_std_px:.3f}px, "
            f"position span={initial.position_span_px:.3f}px, "
            f"center alignment={initial.center_alignment_px:.3f}px "
            f"(stability={initial.center_alignment_std_px:.3f}px), "
            f"radius CV={initial.radius_cv:.3%}, circularity={initial.mean_circularity:.3f}."
        )
        if best.success_rate < 1.0:
            print("Warning: no tested combination passed every calibration frame; "
                  "the best available setting will be used. Inspect the JSON report or "
                  "increase ROI_SIZE / adjust the initial MVGC values.")
        print(f"Adaptive MVGC report: {report_path}\n")
        frame = read_frame(a.input, adaptive.reference_frame, config.io_acceleration)
        if frame is None:
            raise RuntimeError(f"Adaptive reference frame {adaptive.reference_frame} cannot be read")
    pipeline = ConBoPipeline(config)
    if a.show_steps:
        show_method_steps(pipeline, frame, points, a.mode, config)
    if a.mode == "mag":
        print("MAG ROI policy: fixed_initial (the user-selected ROI is identical in every frame; "
              "the adaptive frame supplies the reference phase and zero-displacement position)")
        print(f"MAG algorithm: {config.mag_algorithm}")
    result = pipeline.run(points, a.mode)
    print(f"Completed {a.mode}: {result}")


if __name__ == "__main__":
    main()

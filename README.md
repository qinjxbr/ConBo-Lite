# About ConBo

ConBo (Convolutional Bokeh) is a lightweight vision-based framework for nighttime structural vibration monitoring based on the Bokeh Effect (as seen in the following figure). It transforms luminous targets in bridge videos into shape-normalized bokeh-like responses, enabling stable tracking, displacement extraction, and motion magnification under low-light, low-texture, and long-distance conditions.

<img width="743" height="245" alt="image" src="https://github.com/user-attachments/assets/270757b5-33d4-419c-b5e0-d52b541cb32a" />

Related papers:

1. Jingxi Qin, Mingjin Zhang, Jiale Long, and Wenhui Duan, “Bokeh-based target tracking for structural dynamic monitoring: A novel approach in variable lighting conditions,” *Engineering Structures*, 2024.

2. Jingxi Qin, Jiale Long, Mingjin Zhang, Renan Yuan, Fan Jiang, and Wenhui Duan, “Long-range nighttime dynamic monitoring of long-span cable-stayed bridge with the enhanced bokeh tracking method using closing operation and Long Short-term Memory networks,” *Engineering Structures*, 2025.

3. Jingxi Qin, Wenhui Duan, Jiale Long, Haoliang Zhao, and Yong Xia, “Convolutional Bokeh method for nighttime full-field bridge monitoring via shape-normalized luminous tracking and phase reconstruction-based motion magnification,” *Engineering Structures*, 2026.

# ConBo-Lite

This is a controlled teaching implementation of one 12-step Convolutional Bokeh method with two interfaces. **ConBo-Track is the foundation. ConBo-Mag inherits its target initialization, MVGC preprocessing and detected geometry, then adds fixed-ROI Gaussian phase reconstruction and motion magnification.**

<img width="962" height="546" alt="屏幕截图 2026-07-22 183959" src="https://github.com/user-attachments/assets/3c189e7b-2808-4b80-be11-fdfc38e0e88f" />

## Click Run in an IDE

Open `demo.py`. Its first section contains `RUN_NAME`, `MODE`, input/output paths, frame range, optional points, MVGC, ROI/kernel, amplification, device, video, and display settings. Edit these values and click **Run**. Command-line arguments are optional and override the same settings when supplied.

For normal interactive use, leave `POINTS = None` and click targets in the base frame. Set `POINTS = [(x1, y1), ...]` for repeatable non-interactive runs.

For automatic scene calibration, leave `ADAPTIVE_MVGC = True` directly below `MVGC_GAMMA` and `MVGC_THRESHOLD`. `ADAPTIVE_DURATION_SECONDS` beside the switch controls the calibration duration (`2` gives 40 frames, `1` gives 20, and `0.1` gives 2 at the default sampling rate). The two manual MVGC values become the search seed, not ignored settings. The search prioritizes successful detection and absolute position stability, then center alignment, stable radius, compact foreground, and circularity. Set the switch to `False` to use the two manual values unchanged. A JSON calibration report is saved with the run output.

The default `ADAPTIVE_SEARCH_MODE = "fast"` uses successive halving (4 frames/s, 10 frames/s, then the full random sample) and component-moment proxies to reject weak combinations. The proxy never supplies final coordinates or parameters: sampled finalists and full-interval finalists still run the exact circular convolution and EdgeDrawing path. If a rare full-interval miss remains, a local rescue search is triggered automatically. `"exhaustive"` is retained as a slower comparison mode that applies the exact path to every sampled candidate.

After parameter selection, Adaptive MVGC chooses the position medoid using exact detections from every calibration frame, not only the random sample. `config.base_frame` is updated to this real frame, so Track displacement and MAG phase use it as their zero reference. The user-selected ROI coordinates are deliberately not changed; MAG therefore retains an identical ROI on every frame.

## Installation

```bash
conda env create -f environment.yml
conda activate conbo-lite
```

Or install `requirements.txt` in an existing environment. Use `opencv-contrib-python`, not `opencv-python`, because the primary detector is `cv.ximgproc.createEdgeDrawing()`.

## Run

Mouse selection at native video resolution:

```bash
python demo.py --mode track --input path/to/video.mov --output-dir outputs/track
python demo.py --mode mag --input path/to/video.mov --output-dir outputs/mag --amplification 15
```

Add `--show-steps` to pause on the raw ROI, isolated field, circular Bokeh response, and (for MAG) Gaussian phase-conditioning response. Add `--show-live` for the annotated frame stream.

Non-interactive selection is useful for repeatable runs:

```bash
python demo.py --mode track --input path/to/video.mov --points 820,410 1130,422
python demo.py --mode mag --input path/to/video.mov --points 820,410 1130,422 --device auto
```

Adaptive calibration can also be controlled from the command line:

```bash
python demo.py --mode track --input input/input.mov --points 820,410 --adaptive-mvgc
python demo.py --mode track --input input/input.mov --points 820,410 --no-adaptive-mvgc
```

During mouse selection, click any number of targets, press `U` to undo, and press `N` or Enter to continue. Escape cancels.

## Unified method

1. Read video and base frame.
2. Select luminous targets.
3. Extract boundary-safe local ROIs.
4. Apply MVGC and thresholding.
5. Perform morphological opening.
6. Circular-convolve to create shape-normalized Bokeh responses.
7. Detect EdgeDrawing circles/ellipses and their real radii.
8. Track performs the complete circular-response and EdgeDrawing detection on every target in every frame, then updates target centers and next-frame ROIs. MAG measures the center but retains the initial mouse-selected ROI.
9. For MAG, Gaussian-convolve the same tracked local field.
10. Compute wrapped base/current phase difference.
11. Default MAG amplifies phase, inverse-FFT reconstructs, and detects the amplified ellipse. Optional `phase_shift` applies phase-estimated extra displacement to the currently detected circle without altering raw pixels.
12. Draw detected geometry and save MP4/CSV.

Track uses steps 1–8 and 12. MAG uses all 12; it shares the same preprocessing and detection code but deliberately keeps the base/current ROI coordinates identical. This prevents ROI recentering from contaminating the phase difference used to measure small motion.

## Outputs

Each run always creates a CSV and, when `WRITE_VIDEO = True`, creates `<RUN_NAME>_track.mp4`/`<RUN_NAME>_mag.mp4`. Every processed frame-target pair has one row with the actual ROI center, detected coordinates, displacement, radius, detector method, ROI policy, and the explicit `full_circular_edgedrawing_every_frame` detection policy. `valid` remains the backward-compatible base-detection flag; `track_valid` and `mag_valid` distinguish base-circle success from magnified reconstruction/phase success. `OUTPUT_SCALE` changes only annotated-video resolution; analysis always uses native pixels.

## Parameters

- `roi_size`: local crop size only.
- `gamma`, `threshold`: MVGC isolation controls.
- `adaptive_mvgc`: uses `gamma`/`threshold` as the initial values and calibrates them on a short random sample before processing.
- `adaptive_samples_per_second`, `adaptive_sample_seconds`, `adaptive_sample_start_frame`, `adaptive_random_seed`: control the deterministic per-second stratified sample; partial seconds scale the requested frame count proportionally.
- `adaptive_min_circularity`: minimum roundness accepted during calibration.
- `adaptive_full_validation_candidates`: number of sampled finalists screened by masks on every calibration frame.
- `adaptive_exact_validation_candidates`: number of screened finalists rechecked by the exact convolution/detector on every frame.
- `adaptive_search_mode`: `fast` (successive proxy screening plus exact finals) or `exhaustive` (exact sampled search throughout).
- `morphology_size`, `morphology_shape`: opening size and `ellipse/rect/cross` structure. The default 3x3 ellipse preserves tiny subpixel-shifting spots that a 3x3 rectangle can erase.
- `kernel_size`, `circular_radius_ratio`: Tracking shape-normalization kernel.
- `gaussian_sigma`: MAG spectral-conditioning kernel.
- `amplification_factor`: phase amplification only.
- `mag_algorithm`: `phase_reconstruct` (default IFFT reconstruction and re-detection) or `phase_shift` (fast phase-estimated displacement applied to the currently detected circle).
- `phase_min_confidence`, `phase_max_shift_px`: reject unreliable or implausible fast phase shifts before amplification.
- `mag_reconstruction_padding`: exact MAG FFT padding. `None` automatically adds at least half a kernel on each side and selects an efficient FFT canvas to prevent amplified motion wrapping around the response boundary.
- `update_strategy`: controls Track only. `detected` follows detections; `fixed` keeps selected ROIs. MAG always uses `fixed_initial`.
- `device`: `auto`, `cpu`, or `cuda` for PyTorch convolution/FFT.
- `output_fps`, `codec`: output media settings.
- `io_acceleration`: `cpu` (quiet default), `auto`, or `hardware`; the latter two request OpenCV/FFmpeg hardware I/O and fall back when unavailable.
- `write_video`, `output_scale`: optional CSV-only execution and downscaled annotated video without changing analysis resolution.

Defaults are recorded in `configs/default.yaml`; the executable CLI exposes the main teaching parameters directly.

## Project map

- `conbo/preprocessing.py`: shared MVGC, morphology, kernels, CPU/CUDA convolution.
- `conbo/adaptive_mvgc.py`: short-clip MVGC search, quality scoring, and JSON report.
- `conbo/detection.py`: reusable EdgeDrawing and explicit ellipse fallback.
- `conbo/pipeline.py`: shared target state and 12-step orchestration.
- `conbo/magnification.py`: exact Gaussian-response FFT reconstruction and optional fast phase-shift estimation.
- `conbo/video_io.py`: native-resolution interaction and outputs.
- `docs/`: Chinese tutorials.
- `tests/`: boundary, kernel, and ellipse geometry checks.

## CPU/GPU behavior

The main pipeline's `auto` device mode uses CUDA only when its target batch is large enough. Adaptive's default fast proxy remains on vectorized CPU because the MX450 test showed that many small mask/component operations do not amortize GPU transfer and launch cost; explicitly setting `DEVICE = "cuda"` still enables the experimental GPU proxy. Exact finalist convolution and every-frame Track EdgeDrawing remain unchanged.


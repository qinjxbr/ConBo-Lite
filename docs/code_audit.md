# Controlled refactor audit

## Effective v6.1 flow

`main_6.2_mag_loop.py` and `funcs_loop.py` read one base frame, collect points in a resized window, crop a fixed ROI, apply minimum-channel gamma correction and a threshold, perform morphological opening, place the result on an oversized canvas, convolve it with a circular raised-cosine kernel, and use EdgeDrawing ellipse detection. MAG computes FFT phase difference between base and current response, multiplies it by the amplification factor, reconstructs with inverse FFT, denoises, and detects an ellipse in the amplified response.

## Shared method and the unified 12 steps

Track and MAG share video/base-frame handling, point selection, boundary-safe ROI extraction, MVGC isolation, morphology, target state, ellipse detection, visualization, timing, and CSV/video output. Track completes steps 1–8 and 12 with detected-center ROI updates. MAG reuses the same modules but intentionally fixes all base/current ROIs at the initial click, adds Gaussian spectral conditioning, wrapped phase amplification, and reconstruction in steps 9–11, then uses the same step 12.

## Important differences found in the old lite projects

- Both lite projects replaced EdgeDrawing ellipse detection with an intensity-weighted centroid.
- MAG kept every ROI at the original mouse position instead of using Tracking's detected center.
- Display radii were derived from `circleperi`, not detected ellipse geometry.
- Track and MAG duplicated preprocessing, kernels, interaction, video I/O, and drawing.
- Parameter meaning drifted: `circleperi` represented ROI extent, kernel size, canvas scale, and display diameter in different places.
- The tracking demo added complex ad-hoc prediction/update logic that was not shared by MAG.

## v6.1 defects corrected without changing the method

- Removed hard-coded mouse scale factors (`*2`, `*3.2`, `*3.158`) by selecting at native resolution.
- Made ROI extraction fixed-size and boundary-safe with explicit padding.
- Reused convolution kernels and the EdgeDrawing detector instead of rebuilding them per target/frame.
- Handled zero response and empty ellipse lists without indexing `ellipses[0]`.
- Separated ROI size, kernel size, Gaussian sigma, detected radius, and amplification factor.
- Normalized convolution kernels and responses deterministically; removed the data-dependent `K_max // 1255` scaling.
- Wrapped phase differences to `[-pi, pi]` before amplification to avoid phase-branch jumps.
- Read the base frame once per run and validated video writer/fps/codec.
- Saved one documented CSV row per processed frame and target.

## Preserved algorithmic behavior

MVGC uses the minimum RGB/BGR channel, threshold isolation, morphological opening, circular raised-cosine convolution for shape normalization, EdgeDrawing-based ellipse geometry, base-referenced FFT phase amplification, inverse FFT reconstruction, and ROI-confined output. MAG uses a Gaussian response as described by the manuscript while inheriting Tracking's target state.

## CPU/GPU split

OpenCV/CPU remains responsible for decoding/encoding, morphology, EdgeDrawing or geometric ellipse fallback, GUI selection, and drawing. PyTorch handles reusable convolution and FFT/phase tensor operations. CUDA is selected automatically when available; forcing unavailable CUDA raises an error. This avoids full-frame CPU/GPU transfers because only small ROIs cross the device boundary.

## Intentional scope choices

The executable v6.1 scripts do not implement the manuscript's Kalman/Hungarian association; they process manually selected targets independently. This refactor therefore preserves selected-target identity and detected-center ROI updates rather than inventing a new association algorithm. Adding appearance gates, Kalman prediction, and Hungarian assignment for target crossing or re-entry is a larger method-level extension and is left for review.

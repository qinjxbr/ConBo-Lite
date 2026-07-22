# Validation record

Environment: Python 3.9, OpenCV-contrib 4.10.0 with EdgeDrawing, PyTorch 1.12.0+cu116, NumPy 1.26.4, CUDA available.

## Structural and edge-case checks

- All Python modules compile.
- Boundary ROI at `(1, 2)` retains fixed `20 x 20` shape and correct full-frame mapping.
- Circular and Gaussian kernels sum to one.
- A synthetic filled ellipse is recovered within three pixels and returns detected geometry.
- Empty/low responses return a held state instead of accessing a missing ellipse.

## Existing-video smoke tests

Input: `ConBo_Lite/pnv/input.mov`, 1920x1080, approximately 59.86 fps. Two targets were processed for frames 0–5.

- Tracking CPU: 6 frames, 2 targets, 12 CSV rows. Core preprocessing 0.0477 s; detection 0.0102 s.
- MAG CPU: 6 frames, 2 targets, 12 CSV rows. Preprocessing 0.0396 s; detection 0.0099 s; Gaussian/FFT phase path 0.1789 s.
- MAG CUDA: 6 frames, 2 targets, 12 CSV rows. Preprocessing 0.0401 s; detection 0.0098 s; Gaussian/FFT phase path 0.9110 s.
- Both MP4 outputs reopen successfully, report six 1920x1080 frames, and decode the first frame.

## Fixed-ROI MAG regression

A second two-target, three-frame MAG run audited the recorded ROI center on every CSV row. Target 1 remained exactly `(1162, 44)` and target 2 remained exactly `(1441, 61)` for all frames, with `roi_policy=fixed_initial`. Detected original and amplified centers were free to vary inside those unchanged windows.

For this very short, two-target run, CUDA is slower because startup, kernel launch, and ROI transfer overhead dominate. CUDA is expected to become useful for more/larger ROIs or longer runs; this should be benchmarked on the intended field video rather than assumed.

## Adaptive MVGC regression

Stable case: `input/input2.mov`, point `(1182, 203)`, one second, 20 sampled frames and 30 full-validation frames. Fast search uses vectorized CPU proxy screening and exact finalists.

- The former 3x3 rectangular opening could erase a tiny thresholded spot when its footprint changed by one pixel. The new 3x3 ellipse retains the opening step without requiring a full 3x3 square.
- Fast search selected gamma `2.7`, threshold `93`; exhaustive search selected `2.7/87`. Their exact detection metrics were identical and all 30 frames passed; Fast retained the setting closer to the user's `5.0/100` seed.
- Detected position `(1182.5, 204.5)` had zero x/y span and zero two-dimensional standard deviation; mean circularity was `1.000` and radius CV `0.027%`.
- Every valid frame had zero reference cost, so deterministic tie-breaking retained real frame `0`.
- Fast calibration took `1.22 s`: cache/decode `0.335 s`, proxy plus sampled exact search `0.640 s`, and full validation `0.244 s`.

Moving case: `input/input6.MOV` (119.94 fps), point `(1990, 742)`, two seconds, 40 sampled frames and 240 full-validation frames.

- With initial `5.0/100`, Fast selected gamma `8.778`, threshold `174`; all 240 frames passed, with circularity `0.997` and radius CV `0.239%`.
- Full-frame reference selection corrected the sampled medoid from frame `161` to the true 240-frame medoid, frame `46`; mean/max reference displacement was `10.039/19.776 px`.
- The measured `37.846 px` position span is real trajectory content rather than detector instability: response geometry and foreground isolation remain stable.
- ROI-field caching avoids retaining approximately 1 GB of sampled 4K frames. Fast took `16.06 s`: decode/cache `8.71 s`, search `1.82 s`, and full exact validation including one conditional rescue `5.53 s`. This is about 42% faster than the previous robust `27.63 s` flow while restoring 240/240 success.

## Full Track and output-path regression

- Track explicitly records `detection_policy=full_circular_edgedrawing_every_frame`; its loop generates the circular response and calls the reusable detector once for every target/frame. No moment-center or periodic-recheck Track path exists.
- A three-frame `input2` Track smoke test produced exactly three CSV rows and a reopenable half-resolution MP4 with `OUTPUT_SCALE=0.5`.
- The corresponding `WRITE_VIDEO=False` run produced the same three analysis rows, no MP4 writer, and reported `encode_backend=disabled`.
- OpenCV hardware I/O can be requested, but the current FFmpeg build prints fallback warnings for the tested H.264/HEVC files. `IO_ACCELERATION="cpu"` is therefore the reliable default.

## MAG algorithm modes

- `phase_reconstruct` remains the default and always performs magnified-response detection independently of console-print frequency.
- `phase_shift` retains the current circular-Bokeh detection geometry and applies only the extra `(amplification-1) * phase_shift` to that detected circle; it never warps raw video pixels.
- On five stable `input2.mov` frames, phase work fell from `0.0479 s` to `0.0050 s` (about 9.5x locally); total runtime changed little because decode/write dominated.
- On eleven 4K `input6.MOV` frames, phase work fell from `0.1556 s` to `0.0122 s` (about 12.8x locally), while end-to-end runtime improved about 7%.
- A synthetic translation test recovered `(3.25, -2.5) px` as approximately `(3.20, -2.17) px` with phase-correlation confidence `0.987`.
- CSV rows identify `mag_algorithm`, the phase shift, confidence, and whether geometry came from reconstruction detection or detected-circle-plus-phase-shift.

Full `input2.mov` regression at fixed point `(1182, 205)`, 448 frames, amplification 50, kernel 100:

- Rectangular 3x3 opening erased 200 frames; ellipse 3x3 produced 448/448 base-circle detections, mean circularity above 0.999, and radius CV below 0.2%.
- With no reconstruction padding, exact MAG detected 441/448 magnified responses. The remaining motion was about 2 px, which becomes about 100 px at 50x and exceeded the 159x159 response half-width.
- Automatic FFT padding selected 55/56 pixels around the response (270x270 efficient FFT canvas); exact `phase_reconstruct` then achieved 448/448 `track_valid` and 448/448 `mag_valid` using the parameters selected by the two-second Adaptive run (`3.0/100`).
- `phase_shift` achieved 448/448 in the same full-video regression. A zero-confidence `(80,80)` empty-response result is now rejected before amplification, so invalid phase estimates cannot send the displayed circle outside the frame.

## Remaining validation limits

The original scripts are interactive, use hard-coded resize mappings and unsafe empty-ellipse indexing, so an automated frame-for-frame regression run was not reliable without altering them. The refactor preserves the same effective operations but does not claim pixel-identical responses because kernel normalization, ROI padding, phase wrapping, Gaussian MAG conditioning, and robust ellipse selection intentionally correct known issues. Long field-video accuracy against GPS/laser ground truth was not rerun.

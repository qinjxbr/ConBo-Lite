# About ConBo

ConBo (Convolutional Bokeh) is a lightweight vision-based framework for nighttime structural vibration monitoring based on the Bokeh Effect (as seen in the following figure). It transforms luminous targets in bridge videos into shape-normalized bokeh-like responses, enabling stable tracking, displacement extraction, and motion magnification under low-light, low-texture, and long-distance conditions.

<img width="962" height="317" alt="image" src="https://github.com/user-attachments/assets/270757b5-33d4-419c-b5e0-d52b541cb32a" />

Related papers:

1. Jingxi Qin, Mingjin Zhang, Jiale Long, and Wenhui Duan, “Bokeh-based target tracking for structural dynamic monitoring: A novel approach in variable lighting conditions,” *Engineering Structures*, 2024.

2. Jingxi Qin, Jiale Long, Mingjin Zhang, Renan Yuan, Fan Jiang, and Wenhui Duan, “Long-range nighttime dynamic monitoring of long-span cable-stayed bridge with the enhanced bokeh tracking method using closing operation and Long Short-term Memory networks,” *Engineering Structures*, 2025.

3. Jingxi Qin, Wenhui Duan, Jiale Long, Haoliang Zhao, and Yong Xia, “Convolutional Bokeh method for nighttime full-field bridge monitoring via shape-normalized luminous tracking and phase reconstruction-based motion magnification,” *Engineering Structures*, 2026.

## ConBo-Lite

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

# ConBo Lite 快速使用说明

## 直接在 IDE 中运行

打开项目根目录的 `demo.py`，修改文件最上方的参数区，然后点击 IDE 的 **Run**。

- `RUN_NAME`：输出文件名的前缀。
- `MODE`：`"track"` 为基础跟踪；`"mag"` 为建立在同一跟踪与检测流程上的小位移放大。
- `INPUT_VIDEO`、`OUTPUT_DIR`：输入视频和输出目录。
- `POINTS = None`：运行后用鼠标选点；也可以写成 `[(1990, 742)]` 跳过鼠标步骤。
- `MVGC_GAMMA`、`MVGC_THRESHOLD`：MVGC 初始值。
- `ADAPTIVE_MVGC = True`：自动适配 MVGC；设为 `False` 就完全使用上面两个手动值。
- `ADAPTIVE_DURATION_SECONDS`：校准时长；`2` 表示前两秒共 40 帧，`1` 表示第一秒 20 帧，`0.1` 表示前 0.1 秒抽 2 帧。
- `SHOW_STEPS`：显示原始 ROI → MVGC 前景 → 最终圆形光斑。
- `SHOW_LIVE`：处理过程中显示结果。

## Adaptive MVGC 做什么

选点后、完整处理前，程序会在指定的短时间段内按秒分层随机抽帧。当前默认使用前 1 秒并抽取 20 帧；把 `ADAPTIVE_DURATION_SECONDS` 改成 `2` 就是前两秒共 40 帧，改成 `0.1` 就是前 0.1 秒抽 2 帧。

评分优先保证每个采样帧都能得到有效圆，随后优先比较实际检测位置是否稳定，再比较：

1. 检测坐标的二维标准差和所有采样帧之间的最大位置跨度；
2. MVGC 前景是否为单一干净连通块；
3. 检测圆心是否与该帧前景质心一致；
4. 半径的跨帧变异是否小；
5. 光斑是否接近圆形并远离响应图边界。

默认 `ADAPTIVE_SEARCH_MODE = "fast"`：所有组合先按每秒4帧、前12名按每秒10帧、前8名按全部随机样本逐级筛选；筛选只看MVGC掩膜的连通块、加权中心、半径和二阶矩。代理只负责淘汰，前5名左右仍执行完整圆形卷积和EdgeDrawing。随后前5名在校准区间全部帧做掩膜筛查、前2名做逐帧完整复核；若仍有漏帧，会自动在附近参数做全时段救援。`"exhaustive"` 可用于让每个抽样候选都走完整检测的慢速对照。

最终参数、位置、半径和reference帧都只采用完整圆形响应与EdgeDrawing结果。程序会保存 `<RUN_NAME>_adaptive_mvgc.json`，并分别记录缓存、搜索和验证耗时。

MVGC 参数确定后，程序会在校准时长内的全部真实帧中选择 reference frame：逐一计算每个候选帧到所有位置的二维距离，选择平均总位移最小的“位置中值帧”。随后 Track 的 `dx/dy=0` 和 MAG 的基准相位都以该帧为准。用户最初点击的 ROI 中心不会改变，因此 MAG 每帧仍使用完全相同的固定 ROI。

## 鼠标操作

选择窗口保持原视频宽高比：

- 左键：增加目标；
- `U`：撤销上一个点；
- `N` 或 Enter：完成选点并继续；
- Esc：取消。

## Track 与 MAG

Track 会根据每帧检测位置更新目标（也可把 `TRACK_UPDATE_STRATEGY` 改为 `"fixed"`）。MAG 为保证相位差代表真实小位移，始终在初始点击位置使用完全相同的 ROI，不随每帧检测结果移动。

`MAG_ALGORITHM = "phase_reconstruct"` 是默认方法一致模式：放大频域相位、IFFT 重建空间响应，再从重建结果检测圆。可选的 `"phase_shift"` 快速模式不会扭曲原始帧；当前圆的半径、轴和角度仍来自圆形 Bokeh 检测，只把相位相关得到的额外放大位移施加到这个已检测圆的位置。CSV 会记录 `mag_algorithm`、`phase_dx/dy` 和相位置信度。

微小光斑建议保留 `MORPHOLOGY_SHAPE="ellipse"`：仍是3×3开运算，但不会像矩形结构元素那样要求完整3×3实心块。快速MAG只有在基础圆有效、相位置信度达到`PHASE_MIN_CONFIDENCE`且原始位移不超过`PHASE_MAX_SHIFT_PX`时才应用放大。精确重建模式默认自动给FFT响应补边，避免高放大倍数把目标移出重建画布后发生周期回绕。CSV中的`track_valid`和`mag_valid`可分别定位基础检测与MAG检测问题。

Track 不提供质心/周期复核快模式：每个目标、每一帧都完整生成圆形Bokeh并调用EdgeDrawing。`WRITE_VIDEO=False` 可只输出CSV；`OUTPUT_SCALE=0.5` 只缩小注释视频，检测仍在原始分辨率完成。`IO_ACCELERATION="cpu"` 是当前安静可靠的默认值；安装了可用硬件编解码支持后可试 `"auto"` 或 `"hardware"`。

输出包括标注视频、逐帧 CSV；开启 Adaptive MVGC 时还会多一个 JSON 校准报告。


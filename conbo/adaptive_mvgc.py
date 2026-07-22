"""Scene-adaptive MVGC parameter search before the full ConBo run."""

from dataclasses import asdict, dataclass
from collections import OrderedDict
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import cv2 as cv
import numpy as np

from .config import ConBoConfig
from .detection import EllipseDetector
from .preprocessing import Convolver, extract_padded_roi
from .video_io import open_video_capture


@dataclass
class CalibrationCache:
    """Small per-frame ROI fields retained instead of full decoded video frames."""

    frame_numbers: List[int]
    sample_positions: List[int]
    sample_frames: List[int]
    minimum: np.ndarray
    gray: np.ndarray
    divider: np.ndarray
    anchors: List[Tuple[float, float]]


@dataclass
class CachedPreparedROI:
    isolated: np.ndarray
    response: np.ndarray
    anchor: Tuple[float, float]


@dataclass
class ValidationScreening:
    gamma: float
    threshold: int
    mask_success_rate: float
    exact_evaluated: bool = False
    exact_success_rate: float = 0.0


@dataclass
class ProxyMetrics:
    """Cheap MVGC-mask measurements used only to schedule exact candidates."""

    gamma: float
    threshold: int
    score: float
    success_rate: float
    position_std_px: float
    radius_cv: float
    circularity_proxy: float
    foreground_dominance: float
    foreground_occupancy: float
    border_penalty: float
    observations: int


@dataclass
class CandidateMetrics:
    """Quality measurements for one gamma/threshold combination."""

    gamma: float
    threshold: int
    score: float
    success_rate: float
    position_std_px: float
    position_span_px: float
    center_jitter_px: float
    center_alignment_px: float
    center_alignment_std_px: float
    radius_cv: float
    mean_circularity: float
    circularity_std: float
    foreground_dominance: float
    foreground_occupancy: float
    border_penalty: float


@dataclass
class AdaptiveMVGCResult:
    """Selected parameters and auditable search details."""

    initial_gamma: float
    initial_threshold: int
    initial_reference_frame: int
    gamma: float
    threshold: int
    reference_frame: int
    reference_detected_points: List[Tuple[float, float]]
    reference_mean_displacement_px: float
    reference_max_displacement_px: float
    sample_frames: List[int]
    calibration_frames: int
    elapsed_s: float
    cache_elapsed_s: float
    search_elapsed_s: float
    validation_elapsed_s: float
    observations: int
    candidates_tested: int
    full_validation_candidates: int
    exact_validation_candidates: int
    search_mode: str
    proxy_backend: str
    proxy_candidates_tested: int
    exact_sample_candidates: int
    initial_metrics: CandidateMetrics
    full_validation_screening: List[ValidationScreening]
    sample_top_candidates: List[CandidateMetrics]
    top_candidates: List[CandidateMetrics]


def _sample_frame_numbers(start: int, stop: int, fps: float,
                          samples_per_second: int, seed: int):
    """Return a reproducible per-second stratified sample."""
    rng = np.random.default_rng(seed)
    blocks = []
    block_start = start
    while block_start < stop:
        block_stop = min(stop, block_start + max(1, int(round(fps))))
        available = np.arange(block_start, block_stop, dtype=int)
        block_duration = available.size / fps
        requested = max(1, int(round(samples_per_second * block_duration)))
        count = min(requested, available.size)
        blocks.append(rng.choice(available, size=count, replace=False))
        block_start = block_stop
    return np.sort(np.concatenate(blocks)).astype(int)


def _build_calibration_cache(config: ConBoConfig,
                             points: Sequence[Tuple[int, int]]) -> CalibrationCache:
    """Decode once and cache only minimum-channel and gray target ROIs."""
    cap, _ = open_video_capture(config.input_video, config.io_acceleration)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for adaptive MVGC: {config.input_video}")
    fps = float(cap.get(cv.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    start = max(0, int(config.adaptive_sample_start_frame))
    stop = min(frame_count, start + max(1, int(round(fps * config.adaptive_sample_seconds))))
    if stop <= start:
        cap.release()
        raise RuntimeError("Adaptive MVGC sampling interval contains no readable frames")
    indices = _sample_frame_numbers(start, stop, fps, config.adaptive_samples_per_second,
                                    config.adaptive_random_seed)
    cap.set(cv.CAP_PROP_POS_FRAMES, start)
    frame_numbers, minimum_frames, gray_frames = [], [], []
    anchors = None
    current = start
    while current < stop:
        ok, frame = cap.read()
        if not ok:
            break
        extracted = [extract_padded_roi(frame, point, config.roi_size) for point in points]
        rois = np.stack([item[0] for item in extracted])
        minimum_frames.append(rois.min(axis=3).astype(np.uint8))
        gray_frames.append(np.stack([cv.cvtColor(roi, cv.COLOR_BGR2GRAY) for roi in rois]))
        if anchors is None:
            pad = config.kernel_size // 2
            anchors = [(item[2][0]-pad, item[2][1]-pad) for item in extracted]
        frame_numbers.append(current)
        current += 1
    cap.release()
    if len(frame_numbers) < 2:
        raise RuntimeError(f"Only {len(frame_numbers)} calibration frames could be read")
    position_by_frame = {number: i for i, number in enumerate(frame_numbers)}
    readable_sample_frames = [int(i) for i in indices if int(i) in position_by_frame]
    if len(readable_sample_frames) < 2:
        raise RuntimeError(f"Only {len(readable_sample_frames)} adaptive sample frames could be read")
    sample_positions = [position_by_frame[i] for i in readable_sample_frames]
    minimum = np.stack(minimum_frames)
    divider = minimum.max(axis=(2, 3), keepdims=True).astype(np.float32)
    divider = np.maximum(divider, 1.0)
    return CalibrationCache(frame_numbers, sample_positions, readable_sample_frames,
                            minimum, np.stack(gray_frames), divider, anchors or [])


class AdaptiveIsolationEngine:
    """Cache gamma correction and batch the candidate-by-frame arithmetic."""

    def __init__(self, config: ConBoConfig, cache: CalibrationCache,
                 convolver: Convolver, requested_device: str):
        self.cache = cache
        self.morph_kernel = convolver._morph_kernel
        self.use_cuda = False
        if requested_device == "cuda":
            try:
                import torch
                self.use_cuda = torch.cuda.is_available()
            except ImportError:
                self.use_cuda = False
        self._corrected = OrderedDict()
        self.backend = "cuda_proxy" if self.use_cuda else "vectorized_cpu_proxy"
        minimum_cpu = cache.minimum.astype(np.float32)
        self._normalized_cpu = minimum_cpu/cache.divider
        self._minimum_gpu = self._divider_gpu = None
        if self.use_cuda:
            import torch
            self._minimum_gpu = torch.from_numpy(cache.minimum).to(
                device="cuda", dtype=torch.float32)
            self._divider_gpu = torch.from_numpy(cache.divider).to(
                device="cuda", dtype=torch.float32)

    def corrected(self, gamma: float):
        key = round(float(gamma), 6)
        if key in self._corrected:
            value = self._corrected.pop(key)
            self._corrected[key] = value
            return value
        if self.use_cuda:
            import torch
            with torch.inference_mode():
                value = (torch.pow(self._minimum_gpu/self._divider_gpu, key) *
                         self._divider_gpu).cpu().numpy()
        else:
            value = np.power(self._normalized_cpu, key)*self.cache.divider
        self._corrected[key] = value
        # Bound memory for longer intervals/multiple targets while preserving reuse
        # across the several thresholds tested for the same gamma.
        while len(self._corrected) > 16:
            self._corrected.popitem(last=False)
        return value

    def isolate(self, positions: Sequence[int], gamma: float, threshold: int):
        positions = np.asarray(positions, dtype=int)
        corrected = self.corrected(gamma)[positions]
        gray = self.cache.gray[positions]
        isolated = np.where(corrected > threshold, gray, 0).astype(np.uint8)
        shape = isolated.shape
        flat = isolated.reshape((-1, shape[-2], shape[-1]))
        opened = np.stack([cv.morphologyEx(x, cv.MORPH_OPEN, self.morph_kernel)
                           for x in flat])
        return opened.reshape(shape)


def _isolate_cached_batch(config: ConBoConfig, cache: CalibrationCache, position: int,
                          convolver: Convolver):
    """Apply candidate MVGC and morphology to cached small ROIs."""
    minimum = cache.minimum[position].astype(np.float32)
    divider = cache.divider[position]
    corrected = np.power(minimum / divider, config.gamma) * divider
    isolated = np.where(corrected > config.threshold, cache.gray[position], 0).astype(np.uint8)
    isolated = np.stack([cv.morphologyEx(x, cv.MORPH_OPEN, convolver._morph_kernel)
                         for x in isolated])
    return isolated


def _prepare_cached_batch(config: ConBoConfig, cache: CalibrationCache, position: int,
                          convolver: Convolver):
    """Apply candidate MVGC and exact circular convolution to cached small ROIs."""
    isolated = _isolate_cached_batch(config, cache, position, convolver)
    responses = [convolver._convolve_cpu(x, "circular") for x in isolated]
    return [CachedPreparedROI(isolated[i], responses[i], cache.anchors[i])
            for i in range(len(cache.anchors))]


def _screen_mask_success(config: ConBoConfig, cache: CalibrationCache,
                         gamma: float, threshold: int, engine: AdaptiveIsolationEngine):
    """Cheaply reject candidates that lose or flood targets on any calibration frame."""
    config.gamma, config.threshold = float(gamma), int(threshold)
    success = []
    all_isolated = engine.isolate(range(len(cache.frame_numbers)), gamma, threshold)
    for isolated in all_isolated:
        for target in isolated:
            dominance, occupancy, _ = _foreground_quality(target)
            success.append(dominance >= 0.75 and 0.001 <= occupancy <= 0.45)
    return float(np.mean(success))


def _foreground_quality(isolated: np.ndarray):
    """Measure compactness and return the largest component's weighted centroid."""
    binary = (isolated > 0).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return 0.0, 0.0, None
    count, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0.0, total / binary.size, None
    component = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    largest = int(stats[component, cv.CC_STAT_AREA])
    mask = labels == component
    weights = isolated[mask].astype(np.float64)
    yy, xx = np.nonzero(mask)
    if float(weights.sum()) > 0:
        centroid = (float(np.average(xx, weights=weights)),
                    float(np.average(yy, weights=weights)))
    else:
        centroid = tuple(map(float, centroids[component]))
    return largest / total, total / binary.size, centroid


def _foreground_proxy(isolated: np.ndarray):
    """Return component quality, weighted center, radius, roundness, and border contact."""
    binary = (isolated > 0).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return 0.0, 0.0, None, 0.0, 0.0, 1.0
    count, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0.0, total/binary.size, None, 0.0, 0.0, 1.0
    component = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    area = int(stats[component, cv.CC_STAT_AREA])
    mask = labels == component
    yy, xx = np.nonzero(mask)
    weights = isolated[mask].astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        weights = np.ones_like(xx, dtype=np.float64)
        weight_sum = float(weights.sum())
    cx = float(np.sum(xx*weights)/weight_sum)
    cy = float(np.sum(yy*weights)/weight_sum)
    dx, dy = xx-cx, yy-cy
    covariance = np.array([
        [np.sum(weights*dx*dx)/weight_sum, np.sum(weights*dx*dy)/weight_sum],
        [np.sum(weights*dx*dy)/weight_sum, np.sum(weights*dy*dy)/weight_sum],
    ])
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    circularity = float(np.sqrt(eigenvalues[0]/max(eigenvalues[1], 1e-9)))
    radius = float(np.sqrt(area/np.pi))
    left, top, width, height = stats[component, :4]
    border = float(left <= 0 or top <= 0 or
                   left+width >= isolated.shape[1] or top+height >= isolated.shape[0])
    return area/total, total/binary.size, (cx, cy), radius, circularity, border


def _proxy_evaluate(cache: CalibrationCache, positions: Sequence[int], gamma: float,
                    threshold: int, engine: AdaptiveIsolationEngine) -> ProxyMetrics:
    """Rank masks cheaply; these values never replace exact EdgeDrawing results."""
    isolated = engine.isolate(positions, gamma, threshold)
    targets = isolated.shape[1]
    per_target = [[] for _ in range(targets)]
    success, dominance, occupancy, circularity, border = [], [], [], [], []
    for frame_masks in isolated:
        for target_id, mask in enumerate(frame_masks):
            dom, occ, center, radius, circle, touches = _foreground_proxy(mask)
            good = (center is not None and dom >= 0.75 and 0.001 <= occ <= 0.45 and
                    touches == 0.0)
            success.append(float(good))
            dominance.append(dom)
            occupancy.append(occ)
            circularity.append(circle)
            border.append(touches)
            if good:
                per_target[target_id].append((center[0], center[1], radius))
    position_std, radius_cv = [], []
    for observations in per_target:
        if len(observations) >= 2:
            values = np.asarray(observations, dtype=float)
            position_std.append(float(np.sqrt(np.var(values[:, 0]) +
                                               np.var(values[:, 1]))))
            radius_cv.append(float(np.std(values[:, 2])/max(np.mean(values[:, 2]), 1e-6)))
        else:
            position_std.append(float(cache.minimum.shape[-1]))
            radius_cv.append(1.0)
    success_rate = float(np.mean(success)) if success else 0.0
    mean_position = float(np.mean(position_std))
    mean_radius_cv = float(np.mean(radius_cv))
    mean_circle = float(np.mean(circularity)) if circularity else 0.0
    mean_dom = float(np.mean(dominance)) if dominance else 0.0
    mean_occ = float(np.mean(occupancy)) if occupancy else 0.0
    mean_border = float(np.mean(border)) if border else 1.0
    occupancy_penalty = (max(0.0, 0.002-mean_occ)*100.0 +
                         max(0.0, mean_occ-0.30)*10.0)
    score = ((1.0-success_rate)*1000.0 + mean_position*6.0 +
             mean_radius_cv*120.0 + (1.0-mean_circle)*60.0 +
             (1.0-mean_dom)*25.0 + mean_border*100.0 + occupancy_penalty)
    return ProxyMetrics(float(gamma), int(threshold), float(score), success_rate,
                        mean_position, mean_radius_cv, mean_circle, mean_dom,
                        mean_occ, mean_border, len(success))


def _even_subset(positions: Sequence[int], requested: int):
    """Deterministic temporal coverage for successive-halving stages."""
    positions = list(positions)
    count = min(len(positions), max(2, int(requested)))
    indices = np.unique(np.round(np.linspace(0, len(positions)-1, count)).astype(int))
    return [positions[i] for i in indices]


def _candidate_values(initial_gamma: float, initial_threshold: int):
    """Build a compact coarse grid centered on the user-provided starting values."""
    gamma = sorted({round(initial_gamma, 3)} |
                   {round(float(np.clip(initial_gamma * f, 0.1, 20.0)), 3)
                    for f in (0.60, 0.80, 1.00, 1.25, 1.60)})
    step = max(10, int(round(max(10.0, initial_threshold * 0.25) / 5.0) * 5))
    threshold = sorted({int(np.clip(initial_threshold + step * d, 1, 254))
                        for d in (-2, -1, 0, 1, 2)})
    return gamma, threshold, step


def _evaluate(config: ConBoConfig, cache: CalibrationCache,
              cache_positions: Sequence[int], gamma: float, threshold: int,
              convolver: Convolver, detector: EllipseDetector):
    config.gamma, config.threshold = float(gamma), int(threshold)
    per_target: Dict[int, Dict[str, list]] = {
        i: {"centers": [], "times": [], "radii": [], "circularity": [], "dominance": [],
            "occupancy": [], "alignment": [], "border": [], "success": []}
        for i in range(len(cache.anchors))
    }
    position_trace = []
    for cache_position in cache_positions:
        frame_number = cache.frame_numbers[cache_position]
        prepared = _prepare_cached_batch(config, cache, cache_position, convolver)
        frame_positions = []
        for i, roi in enumerate(prepared):
            q = per_target[i]
            dominance, occupancy, foreground_center = _foreground_quality(roi.isolated)
            q["dominance"].append(dominance)
            q["occupancy"].append(occupancy)
            found = detector.detect(roi.response)
            good = found is not None and found.circularity >= config.adaptive_min_circularity
            if found is not None:
                margin = min(found.center[0]-found.radius, found.center[1]-found.radius,
                             roi.response.shape[1]-1-found.center[0]-found.radius,
                             roi.response.shape[0]-1-found.center[1]-found.radius)
                border_penalty = max(0.0, -margin / max(found.radius, 1.0))
                good = good and border_penalty <= 0.05
                q["centers"].append(found.center)
                frame_positions.append((roi.anchor[0] + found.center[0],
                                        roi.anchor[1] + found.center[1]))
                q["times"].append(frame_number)
                q["radii"].append(found.radius)
                q["circularity"].append(found.circularity)
                q["border"].append(border_penalty)
                if foreground_center is not None:
                    # A symmetric convolution preserves the isolated target's center.
                    # Compare in response coordinates so real inter-frame motion is allowed.
                    pad = config.kernel_size // 2
                    q["alignment"].append(
                        (found.center[0] - (foreground_center[0] + pad),
                         found.center[1] - (foreground_center[1] + pad)))
            else:
                q["border"].append(1.0)
                frame_positions.append(None)
            # Empty, one-pixel, or nearly full masks are not clean target isolation.
            good = good and dominance >= 0.75 and 0.001 <= occupancy <= 0.45
            q["success"].append(bool(good))
            if not good:
                frame_positions[-1] = None
        position_trace.append(frame_positions)

    success, position_std, position_span = [], [], []
    jitter, alignment, alignment_std = [], [], []
    radius_cv, circularity, circ_std = [], [], []
    dominance, occupancy, border = [], [], []
    for q in per_target.values():
        success.append(float(np.mean(q["success"])))
        dominance.append(float(np.mean(q["dominance"])))
        occupancy.append(float(np.mean(q["occupancy"])))
        border.append(float(np.mean(q["border"])))
        if len(q["centers"]) >= 2:
            centers = np.asarray(q["centers"], dtype=float)
            position_std.append(float(np.sqrt(np.var(centers[:, 0]) +
                                               np.var(centers[:, 1]))))
            pairwise = centers[:, None, :] - centers[None, :, :]
            position_span.append(float(np.sqrt(np.sum(pairwise**2, axis=2)).max()))
            times = np.asarray(q["times"], dtype=float)
            if len(times) >= 3 and np.ptp(times) > 0:
                # Remove real constant-velocity motion; score only localization residual.
                x_fit = np.polyval(np.polyfit(times, centers[:, 0], 1), times)
                y_fit = np.polyval(np.polyfit(times, centers[:, 1], 1), times)
                residual_x, residual_y = centers[:, 0]-x_fit, centers[:, 1]-y_fit
                jitter.append(float(np.sqrt(np.var(residual_x) + np.var(residual_y))))
            else:
                jitter.append(float(np.sqrt(np.var(centers[:, 0]) + np.var(centers[:, 1]))))
        else:
            position_std.append(float(config.roi_size))
            position_span.append(float(config.roi_size))
            jitter.append(float(config.roi_size))
        offsets = np.asarray(q["alignment"], dtype=float)
        if offsets.shape[0] >= 2:
            alignment.append(float(np.mean(np.linalg.norm(offsets, axis=1))))
            alignment_std.append(float(np.sqrt(np.var(offsets[:, 0]) +
                                                   np.var(offsets[:, 1]))))
        elif offsets.shape[0] == 1:
            alignment.append(float(np.linalg.norm(offsets[0])))
            alignment_std.append(float(config.roi_size))
        else:
            alignment.append(float(config.roi_size))
            alignment_std.append(float(config.roi_size))
        radii = np.asarray(q["radii"], dtype=float)
        radius_cv.append(float(np.std(radii) / max(np.mean(radii), 1e-6))
                         if radii.size >= 2 else 1.0)
        circles = np.asarray(q["circularity"], dtype=float)
        circularity.append(float(np.mean(circles)) if circles.size else 0.0)
        circ_std.append(float(np.std(circles)) if circles.size >= 2 else 1.0)

    success_rate = float(np.mean(success))
    mean_position_std = float(np.mean(position_std))
    mean_position_span = float(np.mean(position_span))
    center_jitter = float(np.mean(jitter))
    center_alignment = float(np.mean(alignment))
    center_alignment_std = float(np.mean(alignment_std))
    radius_variation = float(np.mean(radius_cv))
    mean_circle = float(np.mean(circularity))
    circle_std = float(np.mean(circ_std))
    mean_dominance = float(np.mean(dominance))
    mean_occupancy = float(np.mean(occupancy))
    mean_border = float(np.mean(border))
    # Failure dominates ranking; remaining terms prefer stable, round, single-component masks.
    occupancy_penalty = (max(0.0, 0.002-mean_occupancy) * 100.0 +
                         max(0.0, mean_occupancy-0.30) * 10.0)
    score = ((1.0-success_rate) * 1000.0 + mean_position_std * 6.0 +
             center_alignment * 3.0 + center_alignment_std * 4.0 +
             center_jitter * 0.5 +
             radius_variation * 120.0 + (1.0-mean_circle) * 60.0 +
             circle_std * 30.0 + (1.0-mean_dominance) * 25.0 +
             mean_border * 100.0 + occupancy_penalty)
    metrics = CandidateMetrics(float(gamma), int(threshold), float(score), success_rate,
                               mean_position_std, mean_position_span, center_jitter,
                               center_alignment, center_alignment_std,
                               radius_variation, mean_circle, circle_std,
                               mean_dominance, mean_occupancy, mean_border)
    return metrics, position_trace


def _select_reference_frame(sample_indices: Sequence[int], position_trace):
    """Choose the real sampled frame minimizing displacement to every observation."""
    valid_frames, positions = [], []
    for frame_number, detected in zip(sample_indices, position_trace):
        if detected and all(p is not None for p in detected):
            valid_frames.append(int(frame_number))
            positions.append(detected)
    if not positions:
        raise RuntimeError("Adaptive MVGC found no fully valid frame for reference selection")
    positions = np.asarray(positions, dtype=float)  # frame, target, xy
    # Candidate i is scored against every observed frame and every target.
    distances = np.linalg.norm(positions[:, None, :, :] - positions[None, :, :, :], axis=3)
    mean_displacement = distances.mean(axis=(1, 2))
    winner = int(np.argmin(mean_displacement))
    reference_points = [tuple(map(float, p)) for p in positions[winner]]
    return (valid_frames[winner], reference_points,
            float(mean_displacement[winner]), float(distances[winner].max()))


def tune_adaptive_mvgc(config: ConBoConfig, points: Sequence[Tuple[int, int]]):
    """Search around the initial settings, update config, and save a JSON audit."""
    started = perf_counter()
    initial_gamma, initial_threshold = float(config.gamma), int(config.threshold)
    initial_reference_frame = int(config.base_frame)
    cache = _build_calibration_cache(config, points)
    cache_finished = perf_counter()
    original_device = config.device
    config.device = "cpu"
    convolver = Convolver(config)
    convolver.configure(len(points), "track")
    isolation_engine = AdaptiveIsolationEngine(config, cache, convolver, original_device)
    detector = EllipseDetector()
    gamma_values, threshold_values, threshold_step = _candidate_values(
        initial_gamma, initial_threshold)
    tested = set()
    proxy_tested = set()
    results = []

    def rank_key(candidate):
        # Deterministic tie-break: prefer the equally good setting nearest the user's seed.
        gamma_distance = abs(candidate.gamma / max(initial_gamma, 1e-6) - 1.0)
        threshold_distance = abs(candidate.threshold - initial_threshold) / max(initial_threshold, 1)
        return round(candidate.score, 6), gamma_distance + threshold_distance

    def run_exact_grid(gammas, thresholds):
        for gamma in gammas:
            for threshold in thresholds:
                key = (round(float(gamma), 3), int(threshold))
                if key in tested:
                    continue
                tested.add(key)
                metrics, _ = _evaluate(config, cache, cache.sample_positions,
                                       key[0], key[1], convolver, detector)
                results.append(metrics)

    half_step = max(2, threshold_step // 2)
    if config.adaptive_search_mode == "exhaustive":
        run_exact_grid(gamma_values, threshold_values)
        # Compatibility mode: every search candidate uses exact convolution/detection.
        for _ in range(3):
            center = min(results, key=rank_key)
            refined_gamma = sorted({round(float(np.clip(center.gamma*f, 0.1, 20.0)), 3)
                                    for f in (0.90, 1.00, 1.10)})
            refined_threshold = sorted({int(np.clip(center.threshold+d*half_step, 1, 254))
                                        for d in (-1, 0, 1)})
            previous_key = (center.gamma, center.threshold)
            run_exact_grid(refined_gamma, refined_threshold)
            updated = min(results, key=rank_key)
            if (updated.gamma, updated.threshold) == previous_key:
                break
    else:
        # Successive halving: use a mask/moment proxy only for scheduling. Every
        # finalist still goes through the original circular response + EdgeDrawing.
        proxy_pool = {}

        def proxy_key(candidate):
            gamma_distance = abs(candidate.gamma/max(initial_gamma, 1e-6)-1.0)
            threshold_distance = abs(candidate.threshold-initial_threshold)/max(initial_threshold, 1)
            return round(candidate.score, 6), gamma_distance+threshold_distance

        def run_proxy(keys, positions):
            evaluated = []
            for gamma, threshold in keys:
                key = (round(float(gamma), 3), int(threshold))
                metrics = _proxy_evaluate(cache, positions, key[0], key[1],
                                          isolation_engine)
                proxy_pool[key] = metrics
                proxy_tested.add(key)
                evaluated.append(metrics)
            return sorted(evaluated, key=proxy_key)

        coarse_keys = [(gamma, threshold) for gamma in gamma_values
                       for threshold in threshold_values]
        duration = max(config.adaptive_sample_seconds, 1.0/max(1.0, 20.0))
        stage1_positions = _even_subset(
            cache.sample_positions, max(2, int(round(duration*4))))
        stage2_positions = _even_subset(
            cache.sample_positions, max(2, int(round(duration*10))))
        stage1 = run_proxy(coarse_keys, stage1_positions)
        stage2_keys = [(x.gamma, x.threshold) for x in stage1[:min(12, len(stage1))]]
        stage2 = run_proxy(stage2_keys, stage2_positions)
        stage3_keys = [(x.gamma, x.threshold) for x in stage2[:min(8, len(stage2))]]
        stage3 = run_proxy(stage3_keys, cache.sample_positions)

        # Explore two small neighborhoods around several survivors, rather than
        # trusting one approximate proxy winner.
        surviving = {(x.gamma, x.threshold): x for x in stage3}
        refinement_step = half_step
        for round_index in range(2):
            centers = sorted(surviving.values(), key=proxy_key)[:3]
            new_keys = set()
            gamma_factor = 0.10 if round_index == 0 else 0.05
            for center in centers:
                for factor in (1.0-gamma_factor, 1.0, 1.0+gamma_factor):
                    for delta in (-refinement_step, 0, refinement_step):
                        new_keys.add((round(float(np.clip(center.gamma*factor, 0.1, 20.0)), 3),
                                      int(np.clip(center.threshold+delta, 1, 254))))
            new_keys -= set(surviving)
            for candidate in run_proxy(sorted(new_keys), cache.sample_positions):
                surviving[(candidate.gamma, candidate.threshold)] = candidate
            refinement_step = max(2, refinement_step//2)

        ranked_proxy = sorted(surviving.values(), key=proxy_key)
        exact_count = max(5, config.adaptive_full_validation_candidates)
        exact_keys = [(x.gamma, x.threshold) for x in ranked_proxy[:exact_count]]
        initial_key = (round(initial_gamma, 3), initial_threshold)
        if initial_key not in exact_keys:
            exact_keys.append(initial_key)
        for gamma, threshold in exact_keys:
            key = (round(float(gamma), 3), int(threshold))
            if key in tested:
                continue
            tested.add(key)
            metrics, _ = _evaluate(config, cache, cache.sample_positions,
                                   key[0], key[1], convolver, detector)
            results.append(metrics)

    results.sort(key=rank_key)
    exact_sample_count = len(results)
    sample_top = results[:8]
    search_finished = perf_counter()
    finalist_count = min(config.adaptive_full_validation_candidates, len(results))
    finalists = results[:finalist_count]
    screening = []
    for candidate in finalists:
        rate = _screen_mask_success(config, cache, candidate.gamma, candidate.threshold,
                                    isolation_engine)
        screening.append(ValidationScreening(candidate.gamma, candidate.threshold, rate))
    screening_by_key = {(round(x.gamma, 3), x.threshold): x for x in screening}
    finalist_by_key = {(round(x.gamma, 3), x.threshold): x for x in finalists}
    ranked_keys = sorted(finalist_by_key, key=lambda key:
                         (-screening_by_key[key].mask_success_rate,
                          rank_key(finalist_by_key[key])))
    full_results, full_traces = [], {}
    all_positions = list(range(len(cache.frame_numbers)))
    exact_target = min(config.adaptive_exact_validation_candidates, len(ranked_keys))
    for key in ranked_keys[:exact_target]:
        metrics, trace = _evaluate(config, cache, all_positions, key[0], key[1],
                                   convolver, detector)
        full_results.append(metrics)
        full_traces[key] = trace
        screening_by_key[key].exact_evaluated = True
        screening_by_key[key].exact_success_rate = metrics.success_rate

    # A rare-frame miss can be invisible in the random sample. If no finalist is
    # perfect, refine locally using every calibration mask, then exact-check only
    # the strongest rescue candidates. This preserves robustness without returning
    # to exhaustive EdgeDrawing for the whole search grid.
    if (config.adaptive_search_mode == "fast" and full_results and
            not any(x.success_rate >= 1.0 for x in full_results)):
        full_results.sort(key=rank_key)
        rescue_step = max(2, half_step//2)
        rescue_keys = set()
        for center in full_results[:3]:
            for factor in (0.95, 1.0, 1.05):
                for delta in (-rescue_step, 0, rescue_step):
                    rescue_keys.add((round(float(np.clip(center.gamma*factor, 0.1, 20.0)), 3),
                                     int(np.clip(center.threshold+delta, 1, 254))))
        # Keep the original unevaluated finalists in the rescue pool too.
        rescue_keys.update(ranked_keys[exact_target:])
        rescue_rank = []
        for key in sorted(rescue_keys):
            if key in screening_by_key:
                rate = screening_by_key[key].mask_success_rate
            else:
                rate = _screen_mask_success(config, cache, key[0], key[1], isolation_engine)
                item = ValidationScreening(key[0], key[1], rate)
                screening.append(item)
                screening_by_key[key] = item
            proxy = _proxy_evaluate(cache, all_positions, key[0], key[1],
                                    isolation_engine)
            seed_distance = (abs(key[0]/max(initial_gamma, 1e-6)-1.0) +
                             abs(key[1]-initial_threshold)/max(initial_threshold, 1))
            rescue_rank.append((-rate, proxy.score, seed_distance, key))
        for _, _, _, key in sorted(rescue_rank)[:6]:
            metrics, trace = _evaluate(config, cache, all_positions, key[0], key[1],
                                       convolver, detector)
            full_results.append(metrics)
            full_traces[key] = trace
            screening_by_key[key].exact_evaluated = True
            screening_by_key[key].exact_success_rate = metrics.success_rate
            if metrics.success_rate >= 1.0:
                break
    initial_key = (round(initial_gamma, 3), initial_threshold)
    if initial_key in full_traces:
        initial_metrics = next(x for x in full_results
                               if x.gamma == initial_key[0] and x.threshold == initial_key[1])
    else:
        initial_metrics, _ = _evaluate(config, cache, all_positions,
                                       initial_key[0], initial_key[1], convolver, detector)
    full_results.sort(key=rank_key)
    best = full_results[0]
    config.gamma, config.threshold = best.gamma, best.threshold
    best_key = (round(float(best.gamma), 3), int(best.threshold))
    reference_frame, reference_points, reference_mean, reference_max = _select_reference_frame(
        cache.frame_numbers, full_traces[best_key])
    config.base_frame = reference_frame
    config.device = original_device
    finished = perf_counter()
    result = AdaptiveMVGCResult(
        initial_gamma, initial_threshold, initial_reference_frame,
        best.gamma, best.threshold, reference_frame, reference_points,
        reference_mean, reference_max,
        cache.sample_frames, len(cache.frame_numbers), finished-started,
        cache_finished-started, search_finished-cache_finished,
        finished-search_finished,
        len(cache.sample_frames)*len(points), len(tested | proxy_tested), len(finalists),
        len(full_results), config.adaptive_search_mode, isolation_engine.backend,
        len(proxy_tested), exact_sample_count,
        initial_metrics, screening, sample_top, full_results[:8])
    report_path = config.output_dir / f"{config.run_name}_adaptive_mvgc.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return result, report_path

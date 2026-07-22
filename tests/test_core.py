import cv2 as cv
import numpy as np
import torch
from pathlib import Path

from conbo.adaptive_mvgc import (_candidate_values, _foreground_quality,
                                 _foreground_proxy, _sample_frame_numbers,
                                 _select_reference_frame)
from conbo.config import ConBoConfig
from conbo.detection import EllipseDetector
from conbo.magnification import BatchPhaseMagnifier, BatchPhaseShiftEstimator
from conbo.pipeline import phase_shift_is_valid
from conbo.preprocessing import Convolver, extract_padded_roi, make_kernel
from conbo.video_io import fit_for_display


def test_border_roi_has_fixed_shape():
    frame = np.full((30, 40, 3), 7, np.uint8)
    roi, bbox, anchor = extract_padded_roi(frame, (1, 2), 20)
    assert roi.shape == (20, 20, 3)
    assert bbox == (0, 0, 11, 12)
    assert anchor == (-9.0, -8.0)


def test_kernels_are_normalized():
    for kind in ("circular", "gaussian"):
        kernel = make_kernel(kind, 31, .45, 6)
        assert np.isclose(kernel.sum(), 1.0, atol=1e-5)


def test_ellipse_detection_reports_geometry():
    image = np.zeros((101, 101), np.uint8)
    cv.ellipse(image, (53, 48), (17, 12), 15, 0, 360, 255, -1)
    found = EllipseDetector().detect(image)
    assert found is not None
    assert abs(found.center[0]-53) < 3 and abs(found.center[1]-48) < 3
    assert found.radius > 8


def test_gaussian_separable_matches_2d_kernel():
    cfg = ConBoConfig(Path("unused"), kernel_size=31, gaussian_sigma=6, device="cpu")
    convolver = Convolver(cfg)
    image = np.zeros((61, 61), np.uint8)
    cv.circle(image, (30, 30), 5, 255, -1)
    fast = convolver._convolve_cpu(image, "gaussian")
    padded = cv.copyMakeBorder(image, 15, 15, 15, 15, cv.BORDER_CONSTANT)
    reference = cv.filter2D(padded.astype(np.float32), cv.CV_32F,
                            convolver._kernels_np["gaussian"],
                            borderType=cv.BORDER_CONSTANT)
    reference = np.clip(reference * (255.0 / reference.max()), 0, 255).astype(np.uint8)
    assert np.max(np.abs(fast.astype(int)-reference.astype(int))) <= 1


def test_full_response_is_not_clipped_and_maps_back():
    cfg = ConBoConfig(Path("unused"), roi_size=60, kernel_size=60,
                      gamma=1.0, threshold=1, morphology_size=3, device="cpu")
    convolver = Convolver(cfg)
    frame = np.zeros((180, 240, 3), np.uint8)
    cv.circle(frame, (120, 90), 4, (255, 255, 255), -1)
    prepared = convolver.prepare(frame, (120, 90), "circular")
    found = EllipseDetector().detect(prepared.response)
    assert prepared.response.shape == (119, 119)
    assert found is not None and found.circularity > .95
    gx, gy = prepared.anchor[0]+found.center[0], prepared.anchor[1]+found.center[1]
    assert abs(gx-120) < 2 and abs(gy-90) < 2


def test_display_fit_preserves_aspect_ratio():
    image = np.zeros((1080, 1920, 3), np.uint8)
    shown, scale = fit_for_display(image, 1000, 700)
    assert shown.shape[:2] == (562, 1000)
    assert np.isclose(shown.shape[1] / shown.shape[0], 1920 / 1080, rtol=.002)


def test_adaptive_grid_contains_user_seed():
    gammas, thresholds, _ = _candidate_values(5.0, 100)
    assert 5.0 in gammas
    assert 100 in thresholds


def test_foreground_quality_uses_largest_component_weighted_center():
    isolated = np.zeros((20, 20), np.uint8)
    isolated[5:9, 7:11] = 200
    isolated[18, 18] = 255
    dominance, occupancy, center = _foreground_quality(isolated)
    assert np.isclose(dominance, 16 / 17)
    assert np.isclose(occupancy, 17 / 400)
    assert np.allclose(center, (8.5, 6.5))


def test_adaptive_proxy_reports_clean_round_component():
    isolated = np.zeros((41, 41), np.uint8)
    cv.circle(isolated, (20, 20), 6, 220, -1)
    dominance, occupancy, center, radius, circularity, border = _foreground_proxy(isolated)
    assert dominance == 1.0
    assert 0.05 < occupancy < 0.1
    assert np.allclose(center, (20, 20))
    assert 5.5 < radius < 6.5
    assert circularity > .99
    assert border == 0.0


def test_ellipse_morphology_preserves_cross_without_square_corners():
    cfg = ConBoConfig(Path("unused"), morphology_size=3,
                      morphology_shape="ellipse", device="cpu")
    kernel = Convolver(cfg)._morph_kernel
    assert kernel[1, 1] == 1
    assert kernel[0, 1] == kernel[1, 0] == kernel[1, 2] == kernel[2, 1] == 1
    assert kernel[0, 0] == kernel[0, 2] == kernel[2, 0] == kernel[2, 2] == 0


def test_phase_shift_guard_rejects_empty_and_implausible_results():
    assert phase_shift_is_valid((0.2, -0.1), .9, .2, 10.0)
    assert not phase_shift_is_valid((80.0, 80.0), 0.0, .2, 10.0)
    assert not phase_shift_is_valid((11.0, 0.0), .9, .2, 10.0)
    assert not phase_shift_is_valid((np.nan, 0.0), .9, .2, 10.0)


def test_phase_reconstruction_padding_expands_fft_canvas():
    base = np.zeros((1, 21, 21), np.float32)
    cv.circle(base[0], (10, 10), 3, 255, -1)
    magnifier = BatchPhaseMagnifier(base, 5.0, torch.device("cpu"), padding=6)
    rebuilt = magnifier.amplify_batch(base)
    assert rebuilt.shape == (1, 33, 33)
    assert rebuilt.max() == 255


def test_adaptive_reference_is_position_medoid():
    frame, points, mean_displacement, max_displacement = _select_reference_frame(
        [0, 1, 2], [[(0.0, 0.0)], [(10.0, 0.0)], [(4.0, 0.0)]])
    assert frame == 2
    assert points == [(4.0, 0.0)]
    assert np.isclose(mean_displacement, 10 / 3)
    assert np.isclose(max_displacement, 6.0)


def test_adaptive_sampling_scales_partial_seconds():
    short = _sample_frame_numbers(0, 3, 30.0, 20, 2026)
    two_seconds = _sample_frame_numbers(0, 60, 30.0, 20, 2026)
    assert len(short) == 2
    assert len(two_seconds) == 40
    assert np.count_nonzero(two_seconds < 30) == 20
    assert np.count_nonzero(two_seconds >= 30) == 20


def test_phase_shift_estimator_reports_translation():
    base = np.zeros((1, 101, 101), np.float32)
    cv.circle(base[0], (50, 50), 8, 1.0, -1)
    transform = np.float32([[1, 0, 3.25], [0, 1, -2.5]])
    current = np.stack([cv.warpAffine(base[0], transform, (101, 101))])
    shifts, confidence = BatchPhaseShiftEstimator(base).estimate_batch(current)
    assert np.allclose(shifts[0], (3.25, -2.5), atol=.4)
    assert confidence[0] > .8

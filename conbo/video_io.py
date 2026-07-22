"""Video, aspect-correct interaction, CSV, trajectory, and drawing helpers."""

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import cv2 as cv
import numpy as np


def open_video_capture(path: Path, acceleration="cpu"):
    """Open a video, trying OpenCV/FFmpeg hardware decode before a safe CPU fallback."""
    if acceleration != "cpu" and all(hasattr(cv, name) for name in
                                      ("CAP_FFMPEG", "CAP_PROP_HW_ACCELERATION",
                                       "VIDEO_ACCELERATION_ANY")):
        cap = cv.VideoCapture(str(path), cv.CAP_FFMPEG,
                              [cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            actual = int(cap.get(cv.CAP_PROP_HW_ACCELERATION))
            return cap, ("opencv_ffmpeg_hardware" if actual else "opencv_ffmpeg")
        cap.release()
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened() and acceleration == "hardware":
        raise RuntimeError(f"Hardware video decode is unavailable for {path}")
    return cap, "opencv_cpu"


def create_video_writer(path: Path, codec: str, fps: float, size, acceleration="cpu"):
    """Create a writer with optional hardware encode and deterministic CPU fallback."""
    fourcc = cv.VideoWriter_fourcc(*codec)
    if acceleration != "cpu" and all(hasattr(cv, name) for name in
                                      ("CAP_FFMPEG", "VIDEOWRITER_PROP_HW_ACCELERATION",
                                       "VIDEO_ACCELERATION_ANY")):
        writer = cv.VideoWriter(str(path), cv.CAP_FFMPEG, fourcc, fps, size,
                                [cv.VIDEOWRITER_PROP_HW_ACCELERATION,
                                 cv.VIDEO_ACCELERATION_ANY])
        if writer.isOpened():
            actual = int(writer.get(cv.VIDEOWRITER_PROP_HW_ACCELERATION))
            return writer, ("opencv_ffmpeg_hardware" if actual else "opencv_ffmpeg")
        writer.release()
    writer = cv.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened() and acceleration == "hardware":
        raise RuntimeError(f"Hardware video encode is unavailable for {path}")
    return writer, "opencv_cpu"


def read_frame(path: Path, index: int, acceleration="cpu"):
    cap, _ = open_video_capture(path, acceleration)
    cap.set(cv.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def fit_for_display(image: np.ndarray, max_width=1600, max_height=900):
    """Fit inside a screen box without ever changing the aspect ratio."""
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                        interpolation=cv.INTER_AREA)
    return resized, scale


def _banner(image: np.ndarray, text: str):
    cv.rectangle(image, (0, 0), (image.shape[1], 38), (20, 20, 20), -1)
    cv.putText(image, text, (12, 26), cv.FONT_HERSHEY_SIMPLEX, .62,
               (255, 255, 255), 1, cv.LINE_AA)


def pick_points(frame: np.ndarray, max_width=1600, max_height=900) -> List[Tuple[int, int]]:
    """Aspect-correct point selection with coordinates mapped to native pixels."""
    shown_base, scale = fit_for_display(frame, max_width, max_height)
    points: List[Tuple[int, int]] = []
    view = shown_base.copy()
    name = "Step 1 - Select targets"

    def redraw():
        view[:] = shown_base
        _banner(view, "Left click: add target   U: undo   N/Enter: confirm   Esc: cancel")
        for i, (x, y) in enumerate(points, 1):
            q = (round(x*scale), round(y*scale))
            cv.circle(view, q, 5, (0, 255, 0), 2)
            cv.putText(view, f"ID{i} ({x},{y})", (q[0]+7, q[1]-7),
                       cv.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1, cv.LINE_AA)

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            native_x = int(np.clip(round(x/scale), 0, frame.shape[1]-1))
            native_y = int(np.clip(round(y/scale), 0, frame.shape[0]-1))
            points.append((native_x, native_y))
            redraw()

    redraw()
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(name, on_mouse)
    while True:
        cv.imshow(name, view)
        key = cv.waitKey(20) & 0xFF
        if key in (ord("n"), ord("N"), 13, 32): break
        if key in (ord("u"), ord("U")) and points:
            points.pop(); redraw()
        if key == 27:
            points.clear(); break
    cv.destroyWindow(name)
    return points


def show_image(window: str, image: np.ndarray, max_width=1600, max_height=900,
               prompt="Space/Enter: continue   Esc: stop tutorial") -> bool:
    """Show an aspect-correct instructional image and return False on Escape."""
    shown, _ = fit_for_display(image, max_width, max_height)
    shown = shown.copy()
    _banner(shown, prompt)
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    cv.imshow(window, shown)
    key = cv.waitKey(0) & 0xFF
    cv.destroyWindow(window)
    return key != 27


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def draw_detection(frame, target_id, center, axes, angle, base_center, quality,
                   coordinates=True):
    """Draw an equivalent detected circle with a readable ID and x/y label."""
    c = tuple(int(round(v)) for v in center)
    radius = max(2, int(round(0.25 * (axes[0] + axes[1]))))
    color = (0, 255, 0) if quality else (0, 165, 255)
    cv.circle(frame, c, radius, color, 2, cv.LINE_AA)
    label = f"ID{target_id}"
    if coordinates:
        label += f"  x={center[0]:.1f}  y={center[1]:.1f}"
    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, .50, 1)
    tx, ty = c[0] + radius + 5, c[1] - radius - 5
    tx = min(max(2, tx), max(2, frame.shape[1]-tw-5))
    ty = min(max(th+5, ty), frame.shape[0]-5)
    cv.rectangle(frame, (tx-3, ty-th-3), (tx+tw+3, ty+4), (0, 0, 0), -1)
    cv.putText(frame, label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, .50,
               color, 1, cv.LINE_AA)
    return frame


def draw_history(frame: np.ndarray, points: Sequence[Tuple[float, float]], color=(255, 200, 0)):
    """Draw a one-second fading trajectory behind the current detection."""
    if len(points) < 2:
        return frame
    pts = np.round(np.asarray(points)).astype(np.int32)
    for i in range(1, len(pts)):
        strength = i / max(1, len(pts)-1)
        segment_color = tuple(int(v * (0.25 + 0.75*strength)) for v in color)
        cv.line(frame, tuple(pts[i-1]), tuple(pts[i]), segment_color,
                max(1, round(1 + strength)), cv.LINE_AA)
    return frame


def overlay_response(frame: np.ndarray, response: np.ndarray, anchor, alpha=.65):
    """Overlay the exact response in its full-frame spatial coordinates."""
    out = frame.copy()
    x0, y0 = map(int, anchor)
    h, w = response.shape
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(frame.shape[1], x0+w), min(frame.shape[0], y0+h)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    crop = response[sy0-y0:sy1-y0, sx0-x0:sx1-x0]
    heat = cv.applyColorMap(crop, cv.COLORMAP_TURBO)
    mask = crop > max(5, int(response.max()*.08))
    roi = out[sy0:sy1, sx0:sx1]
    blended = cv.addWeighted(roi, 1-alpha, heat, alpha, 0)
    roi[mask] = blended[mask]
    return out

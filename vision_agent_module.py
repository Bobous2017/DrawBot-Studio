"""
vision_agent_module.py  –  joint detection using AnimatedDrawings segmentation.
No API key required. Works fully offline.
"""

import cv2
import numpy as np
from skimage import measure
from scipy import ndimage


# --------------------------------------------------------------------------- #
#  Segmentation  (taken from AnimatedDrawings examples/image_to_annotations.py)
# --------------------------------------------------------------------------- #

def _segment(img: np.ndarray) -> np.ndarray:
    """Return a binary mask of the drawn figure."""
    gray = np.min(img, axis=2)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8
    )
    gray = cv2.bitwise_not(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel, iterations=2)

    mask = np.zeros([gray.shape[0] + 2, gray.shape[1] + 2], np.uint8)
    mask[1:-1, 1:-1] = gray.copy()

    im_floodfill = np.full(gray.shape, 255, np.uint8)
    h, w = gray.shape[:2]
    for x in range(0, w - 1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w - 1, y), 0)

    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    mask2 = cv2.bitwise_not(im_floodfill)
    best_mask = None
    biggest = 0
    contours = measure.find_contours(mask2, 0.0)
    for c in contours:
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            best_mask = x
            biggest = size

    if best_mask is None:
        raise ValueError("No figure contour found in image")

    best_mask = ndimage.binary_fill_holes(best_mask).astype(int)
    best_mask = (255 * best_mask).astype(np.uint8)
    return best_mask.T


# --------------------------------------------------------------------------- #
#  Joint estimation from mask geometry
# --------------------------------------------------------------------------- #

def _estimate_joints_from_mask(mask: np.ndarray, img_shape) -> dict:
    """
    Estimate joint positions from the silhouette mask using geometric heuristics.
    Returns joints in the format dance_module expects:
    { "head": [x, y], "neck": [x, y], ... }
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("Empty mask — no figure detected")

    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    h = max_y - min_y
    w = max_x - min_x
    cx = (min_x + max_x) // 2  # horizontal centre

    # --- vertical landmark rows (fractions of figure height) ---
    head_top_y    = min_y
    head_bot_y    = min_y + int(h * 0.22)
    neck_y        = min_y + int(h * 0.26)
    shoulder_y    = min_y + int(h * 0.33)
    elbow_y       = min_y + int(h * 0.50)
    hand_y        = min_y + int(h * 0.65)
    hip_y         = min_y + int(h * 0.60)
    knee_y        = min_y + int(h * 0.78)
    foot_y        = min_y + int(h * 0.95)

    # --- horizontal positions at each row ---
    def _row_span(row_y):
        row_y = max(0, min(mask.shape[0] - 1, row_y))
        cols = np.where(mask[row_y, :] > 0)[0]
        if len(cols) == 0:
            return cx, cx
        return int(cols.min()), int(cols.max())

    head_lx, head_rx = _row_span((head_top_y + head_bot_y) // 2)
    head_cx = (head_lx + head_rx) // 2

    sh_lx, sh_rx = _row_span(shoulder_y)
    el_lx, el_rx = _row_span(elbow_y)
    ha_lx, ha_rx = _row_span(hand_y)
    hi_lx, hi_rx = _row_span(hip_y)
    kn_lx, kn_rx = _row_span(knee_y)
    fo_lx, fo_rx = _row_span(foot_y)

    joints = {
        "head":       [head_cx,             min_y + int(h * 0.10)],
        "neck":       [head_cx,             neck_y],
        "shoulder_l": [sh_lx,               shoulder_y],
        "shoulder_r": [sh_rx,               shoulder_y],
        "elbow_l":    [el_lx,               elbow_y],
        "elbow_r":    [el_rx,               elbow_y],
        "hand_l":     [ha_lx,               hand_y],
        "hand_r":     [ha_rx,               hand_y],
        "hip_l":      [hi_lx + int(w*0.1),  hip_y],
        "hip_r":      [hi_rx - int(w*0.1),  hip_y],
        "knee_l":     [kn_lx + int(w*0.05), knee_y],
        "knee_r":     [kn_rx - int(w*0.05), knee_y],
        "foot_l":     [fo_lx + int(w*0.05), foot_y],
        "foot_r":     [fo_rx - int(w*0.05), foot_y],
    }
    return joints


# --------------------------------------------------------------------------- #
#  Public API  (same signature as before — main.py calls this)
# --------------------------------------------------------------------------- #

def get_robot_joints(image_path: str) -> dict:
    """
    Detect figure joints from image using segmentation only — no API required.

    Returns dict compatible with dance_module:
    { "head": [x, y], "neck": [x, y], ... }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # resize if very large
    if max(img.shape[:2]) > 1000:
        scale = 1000 / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))

    mask = _segment(img)
    joints = _estimate_joints_from_mask(mask, img.shape)
    return joints


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/capture_figure.jpg"
    result = get_robot_joints(path)
    print(json.dumps(result, indent=2))
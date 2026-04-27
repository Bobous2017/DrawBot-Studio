"""
annotate.py  –  Creates AnimatedDrawings-compatible annotations from a photo,
launches the Flask fixer app so joints can be confirmed in the browser,
then runs the animation.

No API key needed. Works fully offline (except the local Flask server).
"""

import os
import sys
import cv2
import numpy as np
import yaml
import base64
import webbrowser
import threading
import time
import shutil
from pathlib import Path
from skimage import measure
from scipy import ndimage

# ── paths ──────────────────────────────────────────────────────────────────────
ROBOT_DIR   = Path(__file__).parent
AD_DIR      = ROBOT_DIR.parent / "AnimatedDrawings"        # sibling folder
EXAMPLES    = AD_DIR / "examples"
FIXER_APP   = EXAMPLES / "fixer_app"
CHAR_DIR    = ROBOT_DIR / "assets" / "char_output"         # where we write files
MOTION_CFG  = (Path("C:/Users/bobx0266/AnimatedDrawings/examples/config/motion/dab.yaml")).resolve()
RETARGET    = (Path("C:/Users/bobx0266/AnimatedDrawings/examples/config/retarget/fair1_ppf.yaml")).resolve()

# ── segmentation (from AnimatedDrawings examples/image_to_annotations.py) ─────

def _segment(img: np.ndarray) -> np.ndarray:
    gray = np.min(img, axis=2)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    gray = cv2.bitwise_not(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,  kernel, iterations=2)
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel, iterations=2)

    mask = np.zeros([gray.shape[0]+2, gray.shape[1]+2], np.uint8)
    mask[1:-1, 1:-1] = gray.copy()
    im_ff = np.full(gray.shape, 255, np.uint8)
    h, w = gray.shape[:2]
    for x in range(0, w-1, 10):
        cv2.floodFill(im_ff, mask, (x, 0),   0)
        cv2.floodFill(im_ff, mask, (x, h-1), 0)
    for y in range(0, h-1, 10):
        cv2.floodFill(im_ff, mask, (0,   y), 0)
        cv2.floodFill(im_ff, mask, (w-1, y), 0)
    im_ff[0,:]=0; im_ff[-1,:]=0; im_ff[:,0]=0; im_ff[:,-1]=0

    mask2 = cv2.bitwise_not(im_ff)
    best, biggest = None, 0
    for c in measure.find_contours(mask2, 0.0):
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            best, biggest = x, size
    if best is None:
        raise ValueError("No figure contour found")
    best = ndimage.binary_fill_holes(best).astype(int)
    return (255 * best).astype(np.uint8).T


def _estimate_joints(mask: np.ndarray, img_shape) -> list:
    """Return skeleton list in AnimatedDrawings char_cfg.yaml format."""
    ys, xs = np.where(mask > 0)
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    h = max_y - min_y
    w = max_x - min_x
    cx = (min_x + max_x) // 2

    def row_span(row_y):
        row_y = max(0, min(mask.shape[0]-1, row_y))
        cols = np.where(mask[row_y, :] > 0)[0]
        return (int(cols.min()), int(cols.max())) if len(cols) else (cx, cx)

    sh_lx, sh_rx = row_span(min_y + int(h * 0.33))
    el_lx, el_rx = row_span(min_y + int(h * 0.50))
    ha_lx, ha_rx = row_span(min_y + int(h * 0.65))
    hi_lx, hi_rx = row_span(min_y + int(h * 0.60))
    kn_lx, kn_rx = row_span(min_y + int(h * 0.78))
    fo_lx, fo_rx = row_span(min_y + int(h * 0.95))

    head_cx = cx
    neck_y  = min_y + int(h * 0.26)

    skeleton = [
        {"name": "root",           "parent": None,            "loc": [cx,                      min_y + int(h*0.60)]},
        {"name": "hip",            "parent": "root",          "loc": [cx,                      min_y + int(h*0.60)]},
        {"name": "torso",          "parent": "hip",           "loc": [cx,                      min_y + int(h*0.33)]},
        {"name": "neck",           "parent": "torso",         "loc": [head_cx,                 neck_y]},
        {"name": "right_shoulder", "parent": "torso",         "loc": [sh_lx,                   min_y + int(h*0.33)]},
        {"name": "right_elbow",    "parent": "right_shoulder","loc": [el_lx,                   min_y + int(h*0.50)]},
        {"name": "right_hand",     "parent": "right_elbow",   "loc": [ha_lx,                   min_y + int(h*0.65)]},
        {"name": "left_shoulder",  "parent": "torso",         "loc": [sh_rx,                   min_y + int(h*0.33)]},
        {"name": "left_elbow",     "parent": "left_shoulder", "loc": [el_rx,                   min_y + int(h*0.50)]},
        {"name": "left_hand",      "parent": "left_elbow",    "loc": [ha_rx,                   min_y + int(h*0.65)]},
        {"name": "right_hip",      "parent": "root",          "loc": [hi_lx + int(w*0.1),      min_y + int(h*0.60)]},
        {"name": "right_knee",     "parent": "right_hip",     "loc": [kn_lx + int(w*0.05),     min_y + int(h*0.78)]},
        {"name": "right_foot",     "parent": "right_knee",    "loc": [fo_lx + int(w*0.05),     min_y + int(h*0.95)]},
        {"name": "left_hip",       "parent": "root",          "loc": [hi_rx - int(w*0.1),      min_y + int(h*0.60)]},
        {"name": "left_knee",      "parent": "left_hip",      "loc": [kn_rx - int(w*0.05),     min_y + int(h*0.78)]},
        {"name": "left_foot",      "parent": "left_knee",     "loc": [fo_rx - int(w*0.05),     min_y + int(h*0.95)]},
    ]
    return skeleton


# ── build char_output folder ───────────────────────────────────────────────────

def build_annotations(image_path: str) -> Path:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    if max(img.shape[:2]) > 1000:
        scale = 1000 / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)))

    CHAR_DIR.mkdir(parents=True, exist_ok=True)

    # texture (RGBA)
    texture = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(str(CHAR_DIR / "texture.png"), texture)

    # mask
    mask = _segment(img)
    cv2.imwrite(str(CHAR_DIR / "mask.png"), mask)

    # skeleton
    skeleton = _estimate_joints(mask, img.shape)
    char_cfg = {
        "skeleton": skeleton,
        "height": img.shape[0],
        "width":  img.shape[1],
    }
    with open(CHAR_DIR / "char_cfg.yaml", "w") as f:
        yaml.dump(char_cfg, f)

    print(f"Annotations saved to: {CHAR_DIR}")
    return CHAR_DIR


# ── Flask fixer app ────────────────────────────────────────────────────────────

def launch_fixer_app(char_dir: Path, port: int = 5050):
    """
    Run AnimatedDrawings' fix_annotations Flask server.
    Opens browser automatically. User clicks Submit when done.
    """
    fix_script = EXAMPLES / "fix_annotations.py"
    if not fix_script.exists():
        print(f"[Warning] fix_annotations.py not found at {fix_script}")
        print("Skipping browser annotation step.")
        return

    import subprocess
    cmd = [
        sys.executable,
        str(fix_script),
        str(char_dir),
        "--port", str(port),
    ]
    print(f"\n=== Opening annotation editor in browser at http://localhost:{port} ===")
    print("Drag the joints to the correct positions, then click Submit.")
    print("Come back here once you've submitted.\n")

    # open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=open_browser, daemon=True).start()

    # run Flask (blocking until user submits or Ctrl+C)
    try:
        subprocess.run(cmd, check=True, cwd=str(EXAMPLES))
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Flask server error: {e}")


# ── AnimatedDrawings animation ─────────────────────────────────────────────────

def animate(char_dir: Path):
    import animated_drawings.render
    ad_examples = Path(r"C:/Users/bobx0266/AnimatedDrawings/examples")
    motion_cfg  = ad_examples / "config" / "motion" / "dab.yaml"
    retarget    = ad_examples / "config" / "retarget" / "fair1_ppf.yaml"
    mvc_cfg = {
        "scene": {
            "ANIMATED_CHARACTERS": [{
                "character_cfg": str((char_dir / "char_cfg.yaml").resolve()),
                "motion_cfg":    str(motion_cfg),
                "retarget_cfg":  str(retarget),
            }]
        },
        "controller": {
            "MODE": "interactive",          # 'interactive' opens a window
        }
    }
    mvc_path = str(char_dir / "mvc_cfg.yaml")
    with open(mvc_path, "w") as f:
        yaml.dump(mvc_cfg, f)

    print("\n=== Starting AnimatedDrawings animation (close window to exit) ===")
    animated_drawings.render.start(mvc_path)


# ── main ───────────────────────────────────────────────────────────────────────

def run(image_path: str):
    print("Step 2: Building annotations …")
    char_dir = build_annotations(image_path)

    print("Step 3: Browser joint editor …")
    launch_fixer_app(char_dir)

    print("Step 4/5: Animating …")
    animate(char_dir)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else str(ROBOT_DIR / "assets" / "capture_figure.jpg")
    run(img)
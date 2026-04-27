"""
app.py  –  Robot Agent Web App
Full pipeline: Capture → Joint Editor → Animate
Run: python app.py
"""

import os
import sys
import cv2
import base64
import json
import yaml
import threading
import time
import subprocess
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from skimage import measure
from scipy import ndimage

# ── paths ──────────────────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
AD_DIR     = Path(r"C:\Users\bobx0266\AnimatedDrawings")
EXAMPLES   = AD_DIR / "examples"
CHAR_DIR   = APP_DIR / "assets" / "char_output"
ASSETS_DIR = APP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
CHAR_DIR.mkdir(parents=True, exist_ok=True)

RETARGET_FAIR1 = str(EXAMPLES / "config" / "retarget" / "fair1_ppf.yaml")
RETARGET_CMU   = str(EXAMPLES / "config" / "retarget" / "cmu1_pfp.yaml")
RETARGET_ROKOKO= str(EXAMPLES / "config" / "retarget" / "fair1_ppf.yaml")

MOTIONS = {
    "Dab":           (str(EXAMPLES / "config" / "motion" / "dab.yaml"),           RETARGET_FAIR1),
    "Jumping Jacks": (str(EXAMPLES / "config" / "motion" / "jumping_jacks.yaml"), RETARGET_CMU),
    "Jumping":       (str(EXAMPLES / "config" / "motion" / "jumping.yaml"),       RETARGET_FAIR1),
    "Wave Hello":    (str(EXAMPLES / "config" / "motion" / "wave_hello.yaml"),    RETARGET_FAIR1),
    "Zombie":        (str(EXAMPLES / "config" / "motion" / "zombie.yaml"),        RETARGET_FAIR1),
}
RETARGET = str(EXAMPLES / "config" / "retarget" / "fair1_ppf.yaml")

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

# ── webcam capture state ───────────────────────────────────────────────────────
capture_state = {"frame": None, "running": False, "captured": False}


# ── segmentation ───────────────────────────────────────────────────────────────
def _segment(img):
    gray = np.min(img, axis=2)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 115, 8)
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
        raise ValueError("No figure found")
    best = ndimage.binary_fill_holes(best).astype(int)
    return (255 * best).astype(np.uint8).T


def _estimate_joints(mask, img_shape):
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

    sh_lx, sh_rx = row_span(min_y + int(h*0.33))
    el_lx, el_rx = row_span(min_y + int(h*0.50))
    ha_lx, ha_rx = row_span(min_y + int(h*0.65))
    hi_lx, hi_rx = row_span(min_y + int(h*0.60))
    kn_lx, kn_rx = row_span(min_y + int(h*0.78))
    fo_lx, fo_rx = row_span(min_y + int(h*0.95))

    return [
        {"name": "root",           "parent": None,             "loc": [cx,                  min_y+int(h*0.60)]},
        {"name": "hip",            "parent": "root",           "loc": [cx,                  min_y+int(h*0.60)]},
        {"name": "torso",          "parent": "hip",            "loc": [cx,                  min_y+int(h*0.33)]},
        {"name": "neck",           "parent": "torso",          "loc": [cx,                  min_y+int(h*0.26)]},
        {"name": "right_shoulder", "parent": "torso",          "loc": [sh_lx,               min_y+int(h*0.33)]},
        {"name": "right_elbow",    "parent": "right_shoulder", "loc": [el_lx,               min_y+int(h*0.50)]},
        {"name": "right_hand",     "parent": "right_elbow",    "loc": [ha_lx,               min_y+int(h*0.65)]},
        {"name": "left_shoulder",  "parent": "torso",          "loc": [sh_rx,               min_y+int(h*0.33)]},
        {"name": "left_elbow",     "parent": "left_shoulder",  "loc": [el_rx,               min_y+int(h*0.50)]},
        {"name": "left_hand",      "parent": "left_elbow",     "loc": [ha_rx,               min_y+int(h*0.65)]},
        {"name": "right_hip",      "parent": "root",           "loc": [hi_lx+int(w*0.1),   min_y+int(h*0.60)]},
        {"name": "right_knee",     "parent": "right_hip",      "loc": [kn_lx+int(w*0.05),  min_y+int(h*0.78)]},
        {"name": "right_foot",     "parent": "right_knee",     "loc": [fo_lx+int(w*0.05),  min_y+int(h*0.95)]},
        {"name": "left_hip",       "parent": "root",           "loc": [hi_rx-int(w*0.1),   min_y+int(h*0.60)]},
        {"name": "left_knee",      "parent": "left_hip",       "loc": [kn_rx-int(w*0.05),  min_y+int(h*0.78)]},
        {"name": "left_foot",      "parent": "left_knee",      "loc": [fo_rx-int(w*0.05),  min_y+int(h*0.95)]},
    ]


def build_annotations(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    if max(img.shape[:2]) > 1000:
        scale = 1000 / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1]*scale), round(img.shape[0]*scale)))
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CHAR_DIR / "texture.png"), cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
    mask = _segment(img)
    cv2.imwrite(str(CHAR_DIR / "mask.png"), mask)
    skeleton = _estimate_joints(mask, img.shape)
    char_cfg = {"skeleton": skeleton, "height": img.shape[0], "width": img.shape[1]}
    with open(CHAR_DIR / "char_cfg.yaml", "w") as f:
        yaml.dump(char_cfg, f)
    return char_cfg


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture")
def capture_page():
    return render_template("capture.html")


@app.route("/api/save_capture", methods=["POST"])
def save_capture():
    data = request.json.get("image", "")
    # data is base64 PNG from browser webcam
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_path = ASSETS_DIR / "capture_figure.jpg"
    cv2.imwrite(str(img_path), img)
    # build annotations
    try:
        char_cfg = build_annotations(img_path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/annotate")
def annotate_page():
    cfg_path = CHAR_DIR / "char_cfg.yaml"
    if not cfg_path.exists():
        return redirect(url_for("capture_page"))
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    texture_path = CHAR_DIR / "texture.png"
    with open(texture_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return render_template("annotate.html", cfg=cfg, image_b64=image_b64)


@app.route("/api/save_joints", methods=["POST"])
def save_joints():
    data = request.json
    cfg_path = CHAR_DIR / "char_cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg["skeleton"] = data["skeleton"]
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return jsonify({"success": True})


@app.route("/animate")
def animate_page():
    return render_template("animate.html", motions=list(MOTIONS.keys()))


@app.route("/api/animate", methods=["POST"])
def do_animate():
    motion_name = request.json.get("motion", "Dab")
    motion_cfg, retarget = MOTIONS.get(motion_name, MOTIONS["Dab"])
    gif_path = CHAR_DIR / "video.gif"

    mvc_cfg = {
        "scene": {
            "ANIMATED_CHARACTERS": [{
                "character_cfg": str((CHAR_DIR / "char_cfg.yaml").resolve()),
                "motion_cfg":    motion_cfg,
                "retarget_cfg":  retarget,
            }]
        },
        "controller": {
            "MODE": "video_render",
            "OUTPUT_VIDEO_PATH": str(gif_path.resolve()),
        }
    }
    mvc_path = str(CHAR_DIR / "mvc_cfg.yaml")
    with open(mvc_path, "w") as f:
        yaml.dump(mvc_cfg, f)

    try:
        import animated_drawings.render
        animated_drawings.render.start(mvc_path)
        return jsonify({"success": True, "gif_url": "/api/gif"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/gif")
def serve_gif():
    gif_path = CHAR_DIR / "video.gif"
    return send_file(str(gif_path), mimetype="image/gif")


@app.route("/api/texture")
def serve_texture():
    return send_file(str(CHAR_DIR / "texture.png"), mimetype="image/png")


if __name__ == "__main__":
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5050")).start()
    app.run(port=5050, debug=False)

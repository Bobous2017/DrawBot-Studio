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


@app.route("/preprocess")
def preprocess_page():
    img_path = ASSETS_DIR / "capture_figure_original.jpg"
    if not img_path.exists():
        img_path = ASSETS_DIR / "capture_figure.jpg"
    if not img_path.exists():
        return redirect(url_for("capture_page"))
    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return render_template("preprocess.html", image_b64=image_b64)

#-------------- Crop Resize -------------------
@app.route("/cropresize")
def cropresize_page():
    img_path = ASSETS_DIR / "capture_figure_original.jpg"
    if not img_path.exists():
        return redirect(url_for("capture_page"))
    return render_template("cropresize.html")

@app.route("/api/save_cropped", methods=["POST"])
def save_cropped():
    data = request.json.get("image", "")
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_path = ASSETS_DIR / "capture_figure.jpg"
    orig_path = ASSETS_DIR / "capture_figure_original.jpg"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(orig_path), img)
    return jsonify({"success": True})

@app.route("/api/original_image")
def original_image():
    img_path = ASSETS_DIR / "capture_figure_original.jpg"
    if not img_path.exists():
        img_path = ASSETS_DIR / "capture_figure.jpg"
    return send_file(str(img_path), mimetype="image/jpeg")

#-------------- Image Process -------------------
@app.route("/api/preview_process", methods=["POST"])
def preview_process():
    data = request.json
    level      = data.get("level", 1)
    threshold  = data.get("threshold", 127)
    block_size = data.get("block_size", 11)
    c_val      = data.get("c_val", 2)
    morph_k    = data.get("morph_k", 2)
    inverted = data.get("proc_inverted", False)

    img_path = ASSETS_DIR / "capture_figure.jpg"
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if level == 1:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    elif level == 2:
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, block_size, c_val)
    else:  # L3
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, block_size, c_val)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if inverted:
        binary = cv2.bitwise_not(binary)

    _, buf = cv2.imencode(".png", binary)
    b64 = base64.b64encode(buf).decode("utf-8")
    return jsonify({"image": "data:image/png;base64," + b64})

@app.route("/api/save_processed", methods=["POST"])
def save_processed():
    import tempfile, subprocess
    data   = request.json
    source = data.get("source", "processed")

    img_path = ASSETS_DIR / "capture_figure.jpg"
    img = cv2.imread(str(img_path))

    if source == "original":
        final = img

    elif source == "processed":
        binary = _apply_processing(data)
        final = binary

    elif source == "cvtrace":
        binary = _apply_processing(data)
        panel_inverted = data.get("inverted", False)
        bm_src = cv2.bitwise_not(binary) if not panel_inverted else binary
        h, w = bm_src.shape
        min_area  = data.get("min_area", 0) * 50
        max_area  = data.get("max_area", 50000)
        epsilon_f = data.get("optimize", 0.2)
        contours, _ = cv2.findContours(bm_src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            epsilon = epsilon_f * 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, max(epsilon, 0.5), True)
            if len(approx) < 3:
                continue
            cv2.fillPoly(canvas, [approx], 0)
        final = canvas

    elif source == "potrace":
        binary = _apply_processing(data)
        panel_inverted = data.get("inverted", False)
        bm_src = cv2.bitwise_not(binary) if not panel_inverted else binary
        tmp_bmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
        tmp_bmp.close()
        tmp_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        tmp_svg.close()
        cv2.imwrite(tmp_bmp.name, bm_src)
        POTRACE = r"C:\Users\bobx0266\Downloads\potrace-1.16.win64\potrace-1.16.win64\potrace.exe"
        subprocess.run([POTRACE, tmp_bmp.name, "-s", "-o", tmp_svg.name,
                        "--turdsize",     str(data.get("turdsize", 500)),
                        "--alphamax",     str(data.get("alphamax", 1.0)),
                        "--opttolerance", str(data.get("opttolerance", 0.2))],
                       capture_output=True)
        os.unlink(tmp_bmp.name)
        tmp_png_path = tmp_svg.name.replace(".svg", ".png")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(tmp_svg.name)
            from PIL import Image as PILImage
            import io
            png_data = renderPM.drawToString(drawing, fmt="PNG")
            arr = np.frombuffer(png_data, dtype=np.uint8)
            final = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if final is None or final.mean() < 1:
                raise ValueError("blank")
            print(f"SVGLIB SUCCESS mean={final.mean():.1f}")
        except Exception as e:
            print(f"SVGLIB FAILED: {e}")
            final = cv2.cvtColor(bm_src, cv2.COLOR_GRAY2BGR)
        try: os.unlink(tmp_svg.name)
        except: pass
        print(f"POTRACE final mean={final.mean():.1f}, shape={final.shape}")  # ← add here

    if len(final.shape) == 2:
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    if source != "original":
        # don't overwrite capture_figure.jpg — keep original for going back
        pass
    else:
        cv2.imwrite(str(img_path), final)

    try:
        if source == "original":
            build_annotations(img_path)
        else:
            img_original = cv2.imread(str(ASSETS_DIR / "capture_figure_original.jpg"))
            if img_original is None:
                img_original = cv2.imread(str(img_path))
            h_orig, w_orig = img_original.shape[:2]
            if max(h_orig, w_orig) > 1000:
                scale = 1000 / max(h_orig, w_orig)
                img_original = cv2.resize(img_original,
                    (round(w_orig*scale), round(h_orig*scale)))

            # resize final to match original dimensions
            final = cv2.resize(final, (img_original.shape[1], img_original.shape[0]))

            CHAR_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(CHAR_DIR / "texture.png"),
                       cv2.cvtColor(final, cv2.COLOR_BGR2BGRA))

            if len(final.shape) == 3:
                final_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            else:
                final_gray = final.copy()

            mean_val = final_gray.mean()
            if mean_val > 127:
                _, mask = cv2.threshold(final_gray, 127, 255, cv2.THRESH_BINARY)
            else:
                _, mask = cv2.threshold(final_gray, 127, 255, cv2.THRESH_BINARY_INV)
            if mask.max() == 0:
                mask = np.ones(final_gray.shape, dtype=np.uint8) * 255

            cv2.imwrite(str(CHAR_DIR / "mask.png"), mask)
            skeleton = _estimate_joints(mask, img_original.shape)
            char_cfg = {
                "skeleton": skeleton,
                "height": img_original.shape[0],
                "width":  img_original.shape[1]
            }
            with open(CHAR_DIR / "char_cfg.yaml", "w") as f:
                yaml.dump(char_cfg, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
@app.route("/api/save_capture", methods=["POST"])
def save_capture():
    data = request.json.get("image", "")
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_path = ASSETS_DIR / "capture_figure.jpg"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(ASSETS_DIR / "capture_figure_original.jpg"), img)
    # ← clear previous session cache
    old = ASSETS_DIR / "last_processed.png"
    if old.exists():
        old.unlink()

    return jsonify({"success": True})

def _apply_processing(data):
    level      = data.get("level", 3)
    threshold  = data.get("threshold", 127)
    block_size = data.get("block_size", 11)
    c_val      = data.get("c_val", 2)
    morph_k    = data.get("morph_k", 2)
    proc_inverted = data.get("proc_inverted", False)  # ← processed panel invert only

    img_path = ASSETS_DIR / "capture_figure_original.jpg"
    if not img_path.exists():
        img_path = ASSETS_DIR / "capture_figure.jpg"

    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if level == 1:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    elif level == 2:
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, block_size, c_val)
    else:
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, block_size, c_val)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if proc_inverted:
        binary = cv2.bitwise_not(binary)
    return binary

@app.route("/api/preview_cvtrace", methods=["POST"])
def preview_cvtrace():
    data        = request.json
    min_area    = data.get("min_area", 0) * 50
    max_area    = data.get("max_area", 50000)
    epsilon_f   = data.get("optimize", 0.2)

    binary = _apply_processing(data)
    panel_inverted = data.get("inverted", False)
    # if panel inverted, lines are black on white → need bitwise_not for findContours
    bm_src = cv2.bitwise_not(binary) if not panel_inverted else binary
    h, w   = bm_src.shape

    contours, _ = cv2.findContours(bm_src, cv2.RETR_CCOMP,
                                    cv2.CHAIN_APPROX_TC89_KCOS)
    paths = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        epsilon = epsilon_f * 0.01 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, max(epsilon, 0.5), True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in pts) + " Z"
        paths.append(f'<path d="{d}" fill="black" stroke="none"/>')

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
           f'<rect width="{w}" height="{h}" fill="white"/>'
           + "".join(paths) + "</svg>")
    b64 = base64.b64encode(svg.encode()).decode()
    return jsonify({"svg": "data:image/svg+xml;base64," + b64})

@app.route("/api/preview_potrace", methods=["POST"])
def preview_potrace():
    import subprocess, tempfile
    data         = request.json
    turdsize     = data.get("turdsize", 500)
    alphamax     = data.get("alphamax", 1.0)
    opttolerance = data.get("opttolerance", 0.2)

    binary = _apply_processing(data)
    panel_inverted = data.get("inverted", False)
    bm_src = cv2.bitwise_not(binary) if not panel_inverted else binary

    tmp_bmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
    tmp_bmp.close()
    tmp_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    tmp_svg.close()
    cv2.imwrite(tmp_bmp.name, bm_src)

    POTRACE = r"C:\Users\bobx0266\Downloads\potrace-1.16.win64\potrace-1.16.win64\potrace.exe"
    result = subprocess.run(
        [POTRACE, tmp_bmp.name, "-s", "-o", tmp_svg.name,
         "--turdsize", str(turdsize),
         "--alphamax", str(alphamax),
         "--opttolerance", str(opttolerance)],
        capture_output=True, text=True)
    os.unlink(tmp_bmp.name)

    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    with open(tmp_svg.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_svg.name)
    return jsonify({"svg": "data:image/svg+xml;base64," + b64})




#-------------- Create Joins and Bones -------------------
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


#-------------- Call animation libraries -------------------
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



#-------------- Download Gif/mp4/ -------------------
@app.route("/api/gif")
def serve_gif():
    gif_path = CHAR_DIR / "video.gif"
    return send_file(str(gif_path), mimetype="image/gif")

@app.route("/api/mp4")
def serve_mp4():
    gif_path = CHAR_DIR / "video.gif"
    mp4_path = CHAR_DIR / "video.mp4"
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", str(gif_path),
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(mp4_path)
        ], check=True, capture_output=True)
        return send_file(str(mp4_path), mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/texture")
def serve_texture():
    return send_file(str(CHAR_DIR / "texture.png"), mimetype="image/png")


if __name__ == "__main__":
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5050")).start()
    app.run(port=5050, debug=False)

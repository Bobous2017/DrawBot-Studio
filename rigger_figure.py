import cv2
import os
import numpy as np


def _build_outer_figure_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, background_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    figure_seed = cv2.bitwise_not(background_mask)

    contours, _ = cv2.findContours(figure_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No drawing detected in image")

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return clean_mask


def remove_background(image_path, output_path=os.path.join("assets", "figure_no_bg.png")):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    clean_mask = _build_outer_figure_mask(img)

    points = cv2.findNonZero(clean_mask)
    if points is None:
        raise ValueError("No drawing detected in image")

    x, y, w, h = cv2.boundingRect(points)
    pad = 10
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1])
    y1 = min(y + h + pad, img.shape[0])
    cropped = img[y0:y1, x0:x1]
    cropped_mask = clean_mask[y0:y1, x0:x1]

    # Apply alpha channel so the result follows the figure shape.
    b, g, r = cv2.split(cropped)
    cutout = cv2.merge([b, g, r, cropped_mask])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, cutout)
    print("Cut figure shape with transparency")
    return output_path


if __name__ == "__main__":
    remove_background(os.path.join("assets", "capture_figure.jpg"))
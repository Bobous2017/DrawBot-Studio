import cv2


def remove_background(image_path, output_path="figure_no_bg.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, thresh])

    cv2.imwrite(output_path, rgba)
    print("Delete background!")
    return output_path


if __name__ == "__main__":
    remove_background("capture_figure.jpg")
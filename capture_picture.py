import cv2


def take_photo():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return None

    print("Press 'Space' to take photo or 'Esc' to exit")

    image_path = "capture_figure.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_path if key == 32 else None


if __name__ == "__main__":
    take_photo()


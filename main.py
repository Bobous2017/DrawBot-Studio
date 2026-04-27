"""
main.py  –  Robot Agent pipeline
Step 1: capture photo from webcam
Steps 2-5: annotate → browser joint editor → AnimatedDrawings animation
"""

import capture_picture
import annotate


def run_robot_agent():
    print("--------- start Robot Agent ---------")

    # Step 1 – capture from webcam
    raw_image = capture_picture.take_photo()
    if not raw_image:
        print("No photo captured. Exiting.")
        return

    # Steps 2-5 – annotate + browser editor + animation
    annotate.run(raw_image)

    print("--------- Robot Agent finished ---------")


if __name__ == "__main__":
    run_robot_agent()
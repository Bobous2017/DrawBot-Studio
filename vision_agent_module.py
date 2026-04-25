import base64
import json
import os

import requests
from dotenv import load_dotenv


load_dotenv()


def get_robot_joints(image_path):
    api_key = os.getenv("Agent_API_Key")
    api_url = os.getenv("Agent_URL")

    if not api_key or not api_url:
        raise ValueError("Agent_API_Key and Agent_URL must be set in the environment")

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = (
        "Identify the joints of this drawing for skeletal rigging. "
        "Return ONLY a JSON object with coordinates (x, y) for: "
        "head, neck, shoulder_l, shoulder_r, elbow_l, elbow_r, hand_l, hand_r, "
        "hip_l, hip_r, knee_l, knee_r, foot_l, foot_r. Use the image dimensions as scale."
    )

    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "google/gemini-flash-3.1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
        },
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    joints_data = result["choices"][0]["message"]["content"]
    if isinstance(joints_data, str):
        return json.loads(joints_data)
    return joints_data


if __name__ == "__main__":
    joints = get_robot_joints("figure_no_bg.png")
    print(json.dumps(joints, indent=4))
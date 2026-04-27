import base64
import json
import os

import requests
from dotenv import load_dotenv


load_dotenv()


def _parse_joints_content(content):
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        raw_text = "\n".join(text_parts).strip()
    else:
        raw_text = str(content).strip()

    if not raw_text:
        raise ValueError("Model returned empty content")

    # Handle markdown-style fenced output and extra explanation text.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])

        preview = raw_text[:300].replace("\n", " ")
        raise ValueError(f"Model response is not valid JSON. Preview: {preview}")


def get_robot_joints(image_path):
    api_key = os.getenv("Agent_API_Key")
    api_url = os.getenv("Agent_URL")
    model_id = os.getenv("Agent_MODEL")

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
            "model": model_id,
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
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        details = response.text
        if response.status_code == 400 and "valid model ID" in details:
            raise RuntimeError(
                f"Invalid model ID: {model_id}. Set Agent_MODEL in .env to a valid model. "
                f"Response: {details}"
            ) from error
        raise RuntimeError(
            f"Request failed: {response.status_code} url={api_url} body={details}"
        ) from error

    result = response.json()
    joints_data = result["choices"][0]["message"]["content"]
    return _parse_joints_content(joints_data)


if __name__ == "__main__":
    joints = get_robot_joints("figure_no_bg.png")
    print(json.dumps(joints, indent=4))
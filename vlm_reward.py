import base64
import math
import re
from pathlib import Path

import cv2
import fire
from openai import OpenAI


def encode_video(path, stride=20):
    v = cv2.VideoCapture(str(path))
    frames, idx = [], 0
    while True:
        ok, frame = v.read()
        if not ok:
            break
        if idx % stride == 0:
            if (frame == 0).all():
                break
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buf).decode())
        idx += 1
    return frames


def predict(video_path, task, n=5):
    p = Path(video_path)
    frames = encode_video(p)
    prompt = f"""
Here is a sequence of frames from a policy rollout video. I need your help determining whether the policy is successful. Does the robot successfully complete the following task? Explain your reasoning, and then output 'Answer: Yes' or 'Answer: No'.
Task: {task}

Note: The Task may have been truncated a bit.
""".strip()
    # The following additional note is required to prevent 4o from
    # too many false negatives. In practice, OpenVLA always picks up
    # the correct object, but 4o struggles with the object identity
    # and often thinks that the incorrect object was picked up.
    prompt += """
Note: Since the video is low resolution, it may be hard to identify specific object identities in from Task description in the video. If the robot completed the task successfully, up to the identity of the objects involved, give the robot the benefit of the doubt.
""".strip()
    client = OpenAI()
    messages = [
        {
            "role": "user",
            "content": [prompt, *[{"image": f, "resize": 256} for f in frames]],
        }
    ]
    res = client.chat.completions.create(model="gpt-4o", messages=messages, n=n)
    votes = sum(1 for c in res.choices if re.search(r"Answer: Yes", c.message.content))
    print("Yes" if votes >= math.ceil(n / 2) else "No")


if __name__ == "__main__":
    fire.Fire(predict)


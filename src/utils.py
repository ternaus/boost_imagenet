from pathlib import Path

import cv2
import numpy as np
import requests


def load_rgb(image_path: Path | str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def download(url, output_path, class_id):
    file_name = url.split("/")[-1]

    output_file_path = output_path / class_id / file_name

    if not output_file_path.exists():
        r = requests.get(url, allow_redirects=True)
        with open(output_file_path, "wb") as file:
            file.write(r.content)

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from getsimilar.get import from_image
from tqdm import tqdm

from src.utils import load_rgb


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path to json with validation.", required=True)
    arg("-im", "--path_to_images", type=Path, help="Path where validation images are saved.", required=True)
    arg("-n", "--num_images", type=int, help="The number of the words images to consider.", default=1000)
    arg("-o", "--output_path", type=Path, help="Path to save json with URLs to similar files.", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.input_path, encoding="utf-8") as file:
        validation_json = json.load(file)

    df = pd.DataFrame(validation_json["losses"]).sort_values(by="loss", ascending=False)

    result = []

    for file_name in tqdm(df["filename"][: args.num_images]):
        image = load_rgb(args.path_to_images / file_name)
        urls = from_image(image, num_similar=20)
        result.append({"filename": file_name, "urls": urls})

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)


if __name__ == "__main__":
    main()

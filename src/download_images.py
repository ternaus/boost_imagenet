import argparse
import json
from pathlib import Path
from typing import Any

import xml2dict
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import download


def get_args() -> Any:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-a", "--json_annotations", type=Path, help="Path to json labels for validation.", required=True)
    arg("-u", "--json_with_urls", type=Path, help="Path to json with similar urls.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save json with URLs to similar files.", required=True)
    arg("-j", "--num_workers", type=int, help="The number of CPU workers", default=4)
    return parser.parse_args()


def main():
    args = get_args()
    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    with open(args.json_with_urls, encoding="utf-8") as file:
        urls = json.load(file)

    results = {}

    for element in tqdm(urls):
        annotation_file_name = Path(element["filename"]).with_suffix(".xml")

        annotation = xml2dict.parse(args.json_annotations / annotation_file_name)["annotation"]["object"]

        if isinstance(annotation, list):
            class_id = annotation[0]["name"]
        else:
            class_id = annotation["name"]

        if class_id in results:
            results[class_id] += element["urls"]
        else:
            results[class_id] = element["urls"]

    for key, value in tqdm(results.items()):
        results[key] = list(set(value))

    for class_id, urls in tqdm(results.items()):
        with open(output_path / f"{class_id}.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(urls))

        output_class_path = output_path / class_id
        output_class_path.mkdir(parents=True, exist_ok=True)

        Parallel(n_jobs=args.num_workers)(delayed(download)(url, output_path, class_id) for url in urls)


if __name__ == "__main__":
    main()

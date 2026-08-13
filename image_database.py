import json
from pathlib import Path


def save_database(data, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


def load_database(input_path):

    input_path = Path(input_path)

    if not input_path.exists():
        return []

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
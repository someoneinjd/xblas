#!/usr/bin/env python3

import sys
import json
import re
from typing import Tuple


def parse(source: str) -> Tuple[str, list[str]]:
    with open(source, "r") as fp:
        code = fp.readlines()
    code = "".join(code).replace("\n", "")
    template_code: str = re.findall(r"template\s*<(.*?)>", code)[-1]
    template_args = [i.strip() for i in template_code.split(",")]
    nontype_args = [
        i for i in template_args if not i.startswith("typename") and "=" not in i
    ]
    return re.findall(r"sycl::event\s+(.*?)\(", code)[-1], [
        i.split(" ")[-1].strip() for i in nontype_args
    ]


def main(files: list[str]):
    if len(files) == 0:
        print("Usage: ./find_all_params.py [file ...]")
        return
    data: dict[str, list[str]] = {}
    for source in files:
        func_name, func_params = parse(source)
        data[func_name] = func_params
    with open("params.json", "w") as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    main(sys.argv[1:])

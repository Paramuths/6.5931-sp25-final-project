import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union
from ruamel.yaml.compat import StringIO
import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.common.nodes import DictNode
from pytimeloop.timeloopfe.v4.art import Art
from pytimeloop.timeloopfe.v4.ert import Ert
import ruamel.yaml
import logging, sys
from numbers import Number
import shutil

logger = logging.getLogger("pytimeloop")
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def show_config(*paths):
    total = ""
    for path in paths:
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            for p in path.glob("*.yaml"):
                with p.open() as f:
                    total += f.read() + "\n"
        else:
            with path.open() as f:
                total += f.read() + "\n"
    print(total)
    # return total

def run_timeloop_model(jinja_parse_data: dict = None, **kwargs):
    if os.path.exists("./output_dir"):
        os.system("rm -r ./output_dir")

    run_accelergy(jinja_parse_data, **kwargs)

    jinja_parse_data = {**(jinja_parse_data or {}), **kwargs}
    spec = tl.Specification.from_yaml_files(
        "designs/top.yaml.jinja2", jinja_parse_data=jinja_parse_data
    )
    spec.ERT = Ert(**DictNode.from_yaml_files("output_dir/ERT.yaml")["ERT"])
    spec.ART = Art(**DictNode.from_yaml_files("output_dir/ART.yaml")["ART"])

    # print(spec)
    # app = tl.to_model_app(spec, output_dir="./output_dir")
    # print('Model app created')
    return tl.call_model(spec, output_dir="./output_dir")


def run_accelergy(jinja_parse_data: dict = None, **kwargs):
    if os.path.exists("./output_dir"):
        os.system("rm -r ./output_dir")
    jinja_parse_data = {**(jinja_parse_data or {}), **kwargs}
    spec = tl.Specification.from_yaml_files(
        "designs/top.yaml.jinja2", jinja_parse_data=jinja_parse_data
    )
    result = tl.accelergy_app(spec, output_dir="./output_dir")

    shutil.copy("output_dir/ART.yaml", "output_dir/timeloop-model.ART.yaml")
    shutil.copy("output_dir/ART.yaml", "output_dir/timeloop-mapper.ART.yaml")
    shutil.copy("output_dir/ERT.yaml", "output_dir/timeloop-model.ERT.yaml")
    shutil.copy("output_dir/ERT.yaml", "output_dir/timeloop-mapper.ERT.yaml")

    return result
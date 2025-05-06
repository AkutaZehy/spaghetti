import argparse
import glob
import multiprocessing as mp
import os
import cv2
import numpy as np
import tqdm
from pathlib import Path
import shutil  # 导入 shutil 模块用于清空目录

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictorm import VisualizationDemo

# constants
WINDOW_NAME = "mask2former demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="A directory containing input images",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations",
        required=True,
    )
    parser.add_argument(
        "--tmp-dir",
        help="A directory to store temporary jpg files",
        default="tmp_jpg",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def convert_bmp_to_jpg(input_path, output_path):
    image = cv2.imread(input_path)
    # print("CONVERT: ", input_path, "->", output_path)
    cv2.imwrite(output_path, image)

def clear_tmp_dir(tmp_dir):
    if os.path.exists(tmp_dir):
        pass

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # 清空临时目录
    clear_tmp_dir(args.tmp_dir)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob('*'):
        if input_file.suffix.lower() == '.bmp':
            jpg_file = Path(args.tmp_dir) / input_file.with_suffix('.jpg').name
            convert_bmp_to_jpg(input_file, jpg_file)
            img = read_image(str(jpg_file), format="BGR")
        elif input_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img = read_image(str(input_file), format="BGR")
        else:
            logger.warning(f"Skipping unsupported file type: {input_file}")
            continue

        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                input_file.name,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        output_file = output_dir / input_file.name
        print("OUTPUT: ", output_file)
        if input_file.suffix.lower() == '.bmp':
            output_file = output_dir / input_file.with_suffix('.jpg').name
        # print(str(output_file))
        visualized_output.save(str(output_file))
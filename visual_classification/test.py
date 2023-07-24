#!/usr/bin/env python3
"""
major actions here: evaluate the trained model
"""
import os
import torch
import warnings

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.models.build_model import build_model
from src.utils.file_io import PathManager

import src.utils.logging as logging
from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    cfg.freeze()
    return cfg


def get_test_loader(cfg, logger):
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        return None
    else:
        return data_loader.construct_test_loader(cfg)


def test(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    test_loader = get_test_loader(cfg, logger)
    if test_loader is None:
        print("No test loader presented. Exit")
        return

    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    # load the saved weights
    model.load_state_dict(torch.load('/home/workspace/chaohao/ggk/TOAST/visual_classification/output/StanfordCars/vit_fb_ppt_small_patch16_224/lr0.01_wd0.0001/run4/test_StanfordCars_logits_100.pth'))

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()

    # evaluation only
    evaluator.evaluate(model, test_loader, "test")


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform testing.
    test(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)

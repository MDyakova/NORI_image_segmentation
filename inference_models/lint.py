""" Code quality evaluation"""

import argparse
import logging
import os
import sys

import pylint.lint

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(prog="LINT")

parser.add_argument(
    "-p",
    "--path",
    help="path to directory you want to run pylint | "
    "Default: %(default)s | "
    "Type: %(type)s ",
    default=os.getcwd(),
    type=str,
)

parser.add_argument(
    "-t",
    "--threshold",
    help="score threshold to fail pylint runner | "
    "Default: %(default)s | "
    "Type: %(type)s ",
    default=7,
    type=float,
)

args = parser.parse_args()
PATH = str(args.path)
threshold = float(args.threshold)

logging.info(
    "PyLint Starting | " "Path: {} | " "Threshold: {} ".format(PATH, threshold)
)

results = pylint.lint.Run(["--disable=E1136, E1137", PATH], do_exit=False)

final_score = results.linter.stats["global_note"]

if final_score < threshold:

    MESSAGE = (
        "PyLint Failed | "
        "Score: {} | "
        "Threshold: {} ".format(final_score, threshold)
    )

    logging.error(MESSAGE)
    raise Exception(MESSAGE)

if final_score >= threshold:
    message = (
        f"PyLint Passed | "
        "Score: {} | "
        "Threshold: {} ".format(final_score, threshold)
    )

    logging.info(message)

    sys.exit(0)
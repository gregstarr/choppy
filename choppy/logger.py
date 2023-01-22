"""Create logger"""
import logging
import sys

import colorlog


class ProgressTracker:
    normal: int = 0
    total_normals: int = 1
    tree: int = 0
    total_trees: int = 1
    part: int = 0
    total_parts: int = 1
    connector_progress: float = 0

    def update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.report()

    @property
    def tree_progress(self):
        prog1 = self.normal / self.total_normals
        prog2 = (prog1 + self.tree) / self.total_trees
        return (prog2 + self.part) / max(self.total_parts, self.part + 1)

    def report(self):
        logger.info("$TREE_PROGRESS %.2f", self.tree_progress)
        logger.info("$CONNECTOR_PROGRESS %.2f", self.connector_progress)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# basic logging setup
stream_formatter = colorlog.ColoredFormatter(
    "[%(asctime)s] %(log_color)s%(levelname)-8s%(reset)s (%(filename)17s:%(lineno)-4s)"
    " %(blue)4s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    },
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

progress = ProgressTracker()
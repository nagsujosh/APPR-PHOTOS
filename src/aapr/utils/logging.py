import logging
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def setup_logger(name: str = "aapr", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_writer(log_dir: str) -> SummaryWriter:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

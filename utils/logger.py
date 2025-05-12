import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = '[%(asctime)s] %(name)s %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
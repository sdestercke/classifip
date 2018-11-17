from . import plot_classification
import logging, sys


def create_logger(name="default", DEBUG=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def is_level_debug(logger):
    return logger.getEffectiveLevel() == logging.DEBUG
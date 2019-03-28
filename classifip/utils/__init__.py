from . import plot_classification


def create_logger(name="default", DEBUG=False):
    import logging, sys
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.flush = sys.stdout.flush
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def is_level_debug(logger):
    import logging
    return logger.getEffectiveLevel() == logging.DEBUG


def normalize_minmax(data_set):
    min_max = list()
    for i in range(len(data_set[0])):
        col_values = [row[i] for row in data_set]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append([value_min, value_max])

    for row in data_set:
        for i in range(len(row)):
            if min_max[i][0] < min_max[i][1]:
                row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
            else:
                row[i] = 1
    return data_set

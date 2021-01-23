from . import plot_classification


def create_logger(name="default", DEBUG=False):
    import logging, sys
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.flush = sys.stdout.flush
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # verify if root logging is turned on
    # @salmuz-bug: generate duplicate line each time
    # if: redirect current logging to sys.stout
    root = logging.getLogger()
    if root.hasHandlers():
        # missing to verify type logging: File/...
        # for handler in root.handlers:
        #     if isinstance(handler, logging.FileHandler):
        #         root.removeHandler(handler)
        while root.hasHandlers():
            root.removeHandler(handler)
        # if len(root.handlers) == 0:
        #     handler = logging.StreamHandler(sys.stdout)
        #     handler.flush = sys.stdout.flush
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     root.addHandler(handler)

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


def timeit(method):
    import time

    def timed(*args, **kwargs):
        DEBUG = args[0].DEBUG if len(args) > 0 and hasattr(args[0], "DEBUG") else True
        if DEBUG:
            ts = time.time()
            result = method(*args, **kwargs)
            te = time.time()
            print(
                "%s - %r  %2.2f ms"
                % (time.strftime("%Y-%m-%d %X"), method.__name__, (te - ts) * 1000)
            )
            return result
        else:
            return method(*args, **kwargs)

    return timed

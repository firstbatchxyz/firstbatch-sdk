import logging


def setup_logger():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger("FirstBatchLogger")
    return logger

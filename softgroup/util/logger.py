import logging

from tensorboardX import SummaryWriter as _SummaryWriter

from .dist import is_main_process, master_only


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('softgroup')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Configure this logger directly so progress output is not affected by
    # whatever logging state a wrapper, notebook, or launcher inherited.
    logger.propagate = False
    effective_level = log_level if is_main_process() else logging.ERROR
    logger.setLevel(effective_level)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(effective_level)
        logger.addHandler(stream_handler)

    if is_main_process() and log_file is not None:
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    return logger


class SummaryWriter(_SummaryWriter):

    @master_only
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    @master_only
    def add_scalar(self, *args, **kwargs):
        return super().add_scalar(*args, **kwargs)

    @master_only
    def flush(self, *args, **kwargs):
        return super().flush(*args, **kwargs)

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This script includes helpful functions to extended PyOpenCl functionality."""

import logging
from functools import partial
from pathlib import Path
from typing import cast


def typed_partial(cls, *args, **kwargs):
    """
    https://stackoverflow.com/questions/61126905/typing-how-to-consider-class-arguments-wrapped-with-partial
    """
    return cast(cls, partial(cls, *args, **kwargs))


def write_string_to_file(data_str: str, file, b_logging: bool=True):
    """

    :param file:
    :param data_str:
    :param relative_folder_path: e.g. relative_folder_path='./results/
    :return:
    """
    path = Path(file).parent
    if not path.exists():
        Path.mkdir(path, parents=True)
        print('Created directory ' + str(path))
    outfile = open(file, 'w')
    if b_logging:
        logging.info('write: ' + str(file))
    outfile.write(data_str)
    outfile.close()

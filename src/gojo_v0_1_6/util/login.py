# Module with login functionalities.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
import warnings
import sys
from datetime import datetime
from loguru import logger


from .validation import checkMultiInputTypes, fileExists


class Login:
    """ Basic Login handler."""
    _is_active = False

    logger_levels = {
        None: logger.info,
        'info': logger.info,
        'error': logger.error,
        'err': logger.error,
        'warn': logger.warning,
        'warning': logger.warning,
        'success': logger.success}


def isActive() -> bool:
    """ Indicates whether the login system is active. """
    return Login._is_active


def deactivate():
    """ Deactivate the current login system. """
    Login._is_active = False


def configureLogger(file: str or None = None, add_time_prefix: bool = True):
    """ Function used to configure the login system. If no file is provided as input the output will be driven by
    the standard Python output. If an input file is provided it will be created and the output will be redirected
    to that file.

    Login levels (when calling the :func:`gojo.io.pprint`):
    - None: Information level
    - 'info': Information level (same as None).
    - 'error': Error level.
    - 'err': Error level (same as 'error')
    - 'warning': Warning level.
    - 'warn': Warning level (same as 'warn').
    - 'success': Successful level.

    Parameters
    ----------
    file : str, default=None
        Output file to redirect the output.

    add_time_prefix : bool, default=True
        Indicate whether to add the time prefix to the login.

    Notes
    -----
    The login status can be checked using :func:`gojo.util.login.isActive`, and can be disabled by using
    :func:`gojo.util.login.deactivate`.
     """
    checkMultiInputTypes(
        ('file', file, [str, type(None)]),
        ('format_file', add_time_prefix, [bool]))

    # add custom handlers
    if file is None:
        logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {level} | {message}", colorize=True)
    else:
        if add_time_prefix:
            file = '{}{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S_'), file.replace('.log', ''))
        fileExists(file, False)
        logger.add(file, format="{time:HH:mm:ss.SS} | {level} | {message}", colorize=True, backtrace=True,
                   diagnose=True)

    Login._is_active = True


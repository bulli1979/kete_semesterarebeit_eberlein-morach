import logging
import os
from typing import Optional


def is_float(value: str):
    """Check if the input value is float.

    :param value: value
    :return: True / False based on whether the value is float or not
    """
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    else:
        return True


def is_int(value: str):
    """Check if the input value is int.

    :param value: value
    :return: True / False based on whether the value is int or not
    """
    try:
        float_value = float(value)
        int_value = int(value)
    except (TypeError, ValueError):
        return False
    else:
        return float_value == int_value


def is_list(value: str):
    """Check if the input value is list.

    Currently, we support list in the following format -

    1. "a,b,c,d"
    2. "1,2,3,4"

    :param value: value
    :return: True / False based on whether the value is list or not
    """
    return "," in value


def is_boolean(value: str):
    """Check if the input value is boolean.

    :param value: value
    :return: True / False based on whether the value is boolean or not
    """
    return value.lower() in ["true", "false"]


def parse_boolean(value: str):
    """Parse the boolean value.

    :param value: value
    :return: Parsed boolean values
    """
    return True if value.lower() == "true" else False


def parse_list(value: str):
    """Parse the list value.

    Currently, we support list in the following format -

    1. "a,b,c,d"
    2. "1,2,3,4"

    :param value: value
    :return: Parsed list
    """
    values = value.split(",")
    clean_values = [v.strip(" \"'") for v in values]
    return [infer_type_and_cast_value(v) for v in clean_values]


def infer_type_and_cast_value(value: Optional[str]):
    """Infer the type of value and casts it accordingly.

    :param value: value
    :return: casted value
    """
    if value is None:
        return value
    elif is_int(value):
        return int(value)
    elif is_float(value):
        return float(value)
    elif is_boolean(value):
        return parse_boolean(value)
    elif is_list(value):
        return parse_list(value)
    else:
        return value


def __setup_fault_handler(file_path: str = None):
    """Set up fault handler.

    :param file_path: path to the error file
    :return:
    """
    try:
        import faulthandler

        if not faulthandler.is_enabled():
            if file_path is not None:
                faulthandler.enable(os.open(file_path, os.O_APPEND), all_threads=True)
            else:
                faulthandler.enable()
    except ImportError:
        logging.warn("No faulthandler found")


def get_error_logger():
    """Return the logger from logging for id ERROR_LOGGER_ID ."""
    return logging.getLogger("error")


def setup_trusted_log(error_volume: str, error_file_path: str):
    """Set up trusted logs for the script.

    :param error_volume: volume where the errors should be written
    :param error_file_path: path to the error_file
    :return: trusted logger
    """
    trusted_log_formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(thread)d %(filename)s:%(lineno)d] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    os.makedirs(error_volume, exist_ok=True)
    trusted_log_handler = logging.FileHandler(error_file_path)
    __setup_fault_handler(file_path=error_file_path)
    trusted_log_handler.setFormatter(trusted_log_formatter)
    trusted_log_handler.setLevel(logging.INFO)

    error_logger = get_error_logger()
    error_logger.addHandler(trusted_log_handler)
    error_logger.propagate = False


def write_trusted_log_info(private_info_message):
    """Write private info message to the trusted log channel.

    :param private_info_message: private trusted log message
    :return:
    """
    trusted_logger = get_error_logger()
    trusted_logger.info(private_info_message)


def write_failure_reason(failure_reason_text, file_path):
    """Write failure reason to failure file.

    :param failure_reason_text: reason for failure
    :param file_path: path to the failure file
    :return:
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        f.write(failure_reason_text)


def write_trusted_log_exception(
    error_message, caused_by, failure_file_path, failure_prefix="Algorithm Error"
):
    """Write private exception message to the trusted error channel.

    :param error_message: error_message
    :param caused_by: cause for the error
    :param failure_file_path: failure file path. Usually /opt/ml/output/failure
    :param failure_prefix: prefix to attach to the error message
    :return:
    """
    message = "{}: {}".format(failure_prefix, error_message)
    error_detail = "Caused by: {}".format(caused_by)
    message += "\n\n{}".format(error_detail)
    err_logger = get_error_logger()
    err_logger.exception(message)
    write_failure_reason(message, failure_file_path)
    return message

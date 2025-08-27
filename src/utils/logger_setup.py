# utils/logger_setup.py

import logging
from colorlog import ColoredFormatter

def setup_logger(level=logging.INFO, log_file=None):
    """
    Настраивает корневой логгер для вывода цветных логов в консоль и
    (опционально) записи в файл.

    :param level: Уровень логирования (например, logging.DEBUG)
    :param log_file: Путь к файлу лога (если не None, логи будут писаться в этот файл)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консольный хендлер с colorlog
    console_handler = logging.StreamHandler()
    log_format = (
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(blue)s%(message)s"
    )
    console_formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Если указан log_file, добавляем файловый хендлер
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger

def color_metric(metric_name, value):
    COLORS = {
        "mF1": "\033[96m",
        "mUAR": "\033[91m",      # бирюзовый / голубой (акустическая нейтральность)
        "ACC": "\033[32m",       # зелёный (интерпретируемо как «норм»)
        "CCC": "\033[33m",       # жёлтый (слегка тревожный — континуальный выход)
        "mean_emo": "\033[1;34m",# жирно-синий (важная агрегированная)
        "mean_pkl": "\033[1;35m" # жирно-фиолетовый (вторая агрегированная)
    }
    END = "\033[0m"
    color = COLORS.get(metric_name, "")
    return f"{color}{metric_name}:{value:.4f}{END}"

def color_split(name: str) -> str:
    SPLIT_COLORS = {
        "TRAIN": "\033[1;33m",  # ярко-жёлтый
        "Dev":   "\033[1;31m",  # ярко-синий
        "Test":  "\033[1;35m",  # ярко-фиолетовый
    }
    END = "\033[0m"
    return f"{SPLIT_COLORS.get(name, '')}{name}{END}"

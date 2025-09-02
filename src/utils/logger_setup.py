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
    """
    Красит ключ-значение метрики. Добавлена поддержка UAR/MF1 и всех recall_*.
    Старые цвета оставлены как есть.
    """
    # базовые ANSI
    END = "\033[0m"
    GRAY = "\033[90m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"

    COLORS = {
        "mF1": "\033[96m",
        "mUAR": "\033[91m",
        "ACC": "\033[32m",
        "CCC": "\033[33m",
        "UAR": "\033[1;34m",
        "MF1": "\033[1;35m",
    }

    # сначала пробуем прямое сопоставление
    color = COLORS.get(metric_name, "")

    # если это per-class recall_* — красим по индексу класса (c0/c1/c2/…)
    if not color and metric_name.lower().startswith("recall_"):
        import re
        m = re.search(r"recall_c(\d+)", metric_name.lower())
        c_idx = int(m.group(1)) if m else None
        if c_idx == 0:
            color = CYAN
        elif c_idx == 1:
            color = YELLOW
        elif c_idx == 2:
            color = MAGENTA
        else:
            color = GRAY

    try:
        return f"{color}{metric_name}:{float(value):.4f}{END}" if color else f"{metric_name}:{float(value):.4f}"
    except Exception:
        # на всякий случай, если value не число
        return f"{color}{metric_name}={value}{END}" if color else f"{metric_name}={value}"


def color_split(name: str) -> str:
    """
    Возвращает раскрашенный тег сплита. Поддерживает верхний регистр:
    TRAIN / DEV / TEST (как у тебя в логах).
    """
    SPLIT_COLORS = {
        "TRAIN": "\033[1;33m",  # ярко-жёлтый
        "DEV":   "\033[1;34m",  # ярко-синий
        "TEST":  "\033[1;35m",  # ярко-фиолетовый
    }
    END = "\033[0m"
    key = name.upper()
    return f"{SPLIT_COLORS.get(key, '')}{name}{END}"

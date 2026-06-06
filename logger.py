import argparse
import logging
import os
import sys
from datetime import datetime


class Logger:
    """
    A singleton logger class to provide a consistent logging setup across the project.
    It configures a root logger that writes to both a file and the console.
    The log file name is automatically generated based on experiment parameters.
    """
    _instance = None
    _initialized = False
    _loggers = {}  # Cache for logger instances by name
    _log_file = None # Stores the generated log file name

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if Logger._initialized:
            return

        Logger._initialized = True

        # Ensure the 'logs' directory exists
        os.makedirs("logs", exist_ok=True)

        # Attempt to parse command-line arguments to create a descriptive log file name
        dataset, method, backbone = self._parse_args()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct a descriptive log file name
        if dataset and method and backbone:
            self.log_file = f"logs/{dataset}_{method}_{backbone}_{current_time}.log"
        else:
            self.log_file = f"logs/experiment_{current_time}.log"

        Logger._log_file = self.log_file

        # Configure the root logger with shared handlers
        self._setup_root_logger()

    def _parse_args(self):
        """
        Parses command-line arguments to get dataset, method, and backbone for the log file name.
        Uses a separate parser to avoid conflicts with the main argument parser.
        """
        parser = argparse.ArgumentParser(description="Logger Argument Parser", add_help=False)
        parser.add_argument("--dataset", type=str, default="unknown_dataset")
        parser.add_argument("--method", type=str, default="unknown_method")
        parser.add_argument("--backbone", type=str, default="unknown_backbone")
        try:
            args, _ = parser.parse_known_args()
            return args.dataset, args.method, args.backbone
        except Exception:
            # In case of any parsing error, return default values
            return "unknown", "unknown", "unknown"

    def _setup_root_logger(self):
        """
        Sets up the root logger. All loggers created by get_logger will inherit this configuration.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers to avoid duplicate logs
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls, name="Experiment"):
        """
        Gets a logger instance with a specific name.
        If the logger does not exist, it is created and cached.
        """
        if not cls._initialized:
            cls() # Initialize the singleton if not already done

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        # The logger will use the handlers configured on the root logger
        logger.propagate = True
        cls._loggers[name] = logger
        return logger

    @classmethod
    def get_log_file(cls):
        """Returns the path to the current log file."""
        if not cls._initialized:
            cls()
        return cls._log_file


# Example usage
if __name__ == "__main__":
    # The logger is automatically configured on the first call to get_logger
    main_logger = Logger.get_logger("Main")
    data_logger = Logger.get_logger("DataLoader")
    trainer_logger = Logger.get_logger("Trainer")

    main_logger.debug("This is a debug message from the Main logger.")
    data_logger.info("This is an info message from the DataLoader logger.")
    trainer_logger.warning("This is a warning message from the Trainer logger.")
    main_logger.error("This is an error message from the Main logger.")
    data_logger.critical("This is a critical message from the DataLoader logger.")
    
    print(f"Log file is located at: {Logger.get_log_file()}")

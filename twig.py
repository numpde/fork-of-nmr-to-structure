import logging
import os

# Define ANSI color codes for different parts of the log message

COLORS_HI = {
    'asctime': '\033[1;36m',  # Cyan
    'name': '\033[1;32m',  # Green for logger name
    'module': '\033[1;34m',  # Blue for module
    'levelname': {
        'D': '\033[0;32m',  # Dull green for DEBUG
        'I': '\033[1;32m',  # Bright green for INFO
        'W': '\033[1;33m',  # Bright yellow for WARNING
        'E': '\033[0;31m',  # Red for ERROR
        'C': '\033[1;31m'  # Bold red for CRITICAL
    },
    'message': '\033[1;35m',  # Magenta for the log message itself
    'reset': '\033[0m'  # Reset color
}

COLORS_LO = {
    'asctime': '\033[0;37m',  # Light gray for timestamp
    'name': '\033[0;90m',  # Gray for logger name
    'module': '\033[0;90m',  # Gray for module
    'levelname': {
        'D': '\033[0;90m',  # Gray for DEBUG
        'I': '\033[0;32m',  # Dull green for INFO
        'W': '\033[0;33m',  # Dull yellow for WARNING
        'E': '\033[0;31m',  # Dim red for ERROR
        'C': '\033[0;35m'  # Dim magenta for CRITICAL
    },
    'message': '\033[0;37m',  # Light gray for the log message
    'reset': '\033[0m'  # Reset color
}

LEVEL_ABBREVIATIONS = {
    'DEBUG': 'D',
    'INFO': 'I',
    'WARNING': 'W',
    'ERROR': 'E',
    'CRITICAL': 'C'
}


class CustomFormatter(logging.Formatter):
    def __init__(self, hi=False):
        super().__init__()
        self.COLORS = COLORS_HI if hi else COLORS_LO

    def format(self, record):
        # Extract parent folder and module name
        filepath = record.pathname
        parent_folder, module = os.path.split(filepath)
        parent_folder_name = os.path.basename(parent_folder)
        record.module = f"{parent_folder_name}/{module.replace('.py', '')}"

        # Abbreviate and color the level name
        level_abbr = LEVEL_ABBREVIATIONS.get(record.levelname, record.levelname[0])
        level_color = self.COLORS['levelname'].get(level_abbr, self.COLORS['reset'])
        record.levelname = f"{level_color}{level_abbr}{self.COLORS['reset']}"

        # Color the message itself based on the log level
        message_color = self.COLORS['levelname'].get(level_abbr, self.COLORS['reset'])
        formatted_message = f"{message_color}{record.getMessage()}{self.COLORS['reset']}"

        # Build the log message with the colored components
        log_message = (
            f"{self.COLORS['asctime']}{self.formatTime(record)}{self.COLORS['reset']} "
            f"{record.levelname} "
            f"{formatted_message}"
            f"  "
            f"{self.COLORS['name']}@{record.name}/{self.COLORS['reset']}"
            f"{self.COLORS['module']}{record.module}{self.COLORS['reset']} "
        )

        return log_message

    def formatTime(self, record, datefmt=None):
        # Format time without milliseconds and use custom format
        datefmt = "%Y%m%d %H%M%S"
        return super().formatTime(record, datefmt)


logging.basicConfig(
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Apply the custom formatter to all handlers
for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter())

log = logging.getLogger(__name__)

if __name__ == '__main__':
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")

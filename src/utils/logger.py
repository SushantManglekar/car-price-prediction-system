import os
import argparse
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class Logger:
    LEVELS = {
        'INFO': (Fore.GREEN, "INFO"),
        'DEBUG': (Fore.CYAN, "DEBUG"),
        'WARNING': (Fore.YELLOW, "WARNING"),
        'ERROR': (Fore.RED, "ERROR")
    }

    def __init__(self, level='INFO', disable_warning=False, disable_error=False, write_logs=True):
        self.level = level
        self.disable_warning = disable_warning
        self.disable_error = disable_error
        self.write_logs = write_logs
        
        # Get the root directory of the project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # Define the logs folder path in the root directory
        self.log_dir = os.path.join(project_root, 'logs')

        if self.write_logs:
            # Create logs folder if it doesn't exist
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)  # Create logs folder if not exists

            # Create log file name based on current date and time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_file = os.path.join(self.log_dir, f"{timestamp}.log")

    def _write_message(self, message):
        if self.write_logs:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

    def _format_message(self, level, message):
        color, level_name = self.LEVELS.get(level, (Style.RESET_ALL, "UNKNOWN"))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {level_name} | {message}"

        if self.write_logs:
            return formatted_message
        else:
            return f"{color}{formatted_message}{Style.RESET_ALL}"

    def log_debug(self, message):
        if self.level == 'DEBUG':
            formatted_message = self._format_message('DEBUG', message)
            self._write_message(formatted_message)

    def log_info(self, message):
        if self.level in ['DEBUG', 'INFO']:
            formatted_message = self._format_message('INFO', message)
            self._write_message(formatted_message)

    def log_warning(self, message):
        if self.level in ['DEBUG', 'INFO', 'WARNING'] and not self.disable_warning:
            formatted_message = self._format_message('WARNING', message)
            self._write_message(formatted_message)

    def log_error(self, message):
        if self.level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'] and not self.disable_error:
            formatted_message = self._format_message('ERROR', message)
            self._write_message(formatted_message)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Custom Logging Program")
    parser.add_argument('--debug', '-d', action='store_true', help="Enable debug logging")
    parser.add_argument('--disable-warning', '-w', action='store_true', help="Disable warning logging")
    parser.add_argument('--disable-error', '-e', action='store_true', help="Disable error logging")
    parser.add_argument('--logfile', type=str, help="Log file path (optional)")
    return parser.parse_args()

def create_logger():
    args = parse_arguments()
    level = 'DEBUG' if args.debug else 'INFO'
    return Logger(
        level,
        disable_warning=args.disable_warning,
        disable_error=args.disable_error,
        write_logs=True
    )

if __name__ == "__main__":
    logger = create_logger()
    logger.log_info("Logger initialized.")

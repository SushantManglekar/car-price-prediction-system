import sys

class CustomException(Exception):
    """Custom exception that includes an error message, file name, and line number where the exception was raised."""

    def __init__(self, error_message):
        # Get the caller’s frame (1 means the caller’s frame, 0 would be the current frame)
        frame = sys._getframe(1)
        file_name = frame.f_code.co_filename
        line_number = frame.f_lineno
        
        # Construct the full error message with file name and line number
        self.error_message = f"{error_message} (File: {file_name}, Line: {line_number})"
        super().__init__(self.error_message)

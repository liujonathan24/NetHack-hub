"""Inert stand-in for ``netplay.logging.step_logger.StepLogger``."""


class StepLogger:
    def __init__(self, log_folder=None):
        self.log_folder = log_folder

    def start_next_step(self):
        pass

    def log_json(self, data=None, file_name=None):
        pass

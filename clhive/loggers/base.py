from abc import abstractmethod, ABC
from typing import Any


class Logger(ABC):
    @abstractmethod
    def __init__(
        self,
        log_dir: str,
    ):
        self.log_dir = log_dir

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def log_dict(self):
        pass

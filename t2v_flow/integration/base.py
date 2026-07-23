# my_system/base.py

from abc import ABC, abstractmethod

class CustomSystem(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """

        Returns:
        """
        pass

    @abstractmethod
    def initialize(self, args):
        """
        

        Args:
        """
        pass
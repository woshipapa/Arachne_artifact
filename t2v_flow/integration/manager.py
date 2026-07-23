# my_system/manager.py

import threading
from typing import List
from .base import CustomSystem

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class SystemManager(metaclass=SingletonMeta):
    def __init__(self):
        self._systems: List[CustomSystem] = []
        print("SystemManager Singleton initialized.")

    def register(self, system: CustomSystem):
        """

        Args:

        Raises:
        """
        if not isinstance(system, CustomSystem):
            raise TypeError(
                f"Can only register objects of type CustomSystem, "
                f"but got {type(system).__name__}"
            )
        
        print(f"  -> System '{system.name}' registered with the manager.")
        self._systems.append(system)

    def initialize_all(self, args):
        """

        Args:
        """
        print("\n--- SystemManager: Initializing all custom systems... ---")
        if not self._systems:
            print("No custom systems registered. Skipping.")
            return
        
        for system in self._systems:
            try:
                print(f"Initializing system: '{system.name}'...")
                system.initialize(args)
            except Exception as e:
                print(f"\nFATAL: Failed to initialize system '{system.name}'.")
                print(f"Error: {e}\n")
                raise
                
        print("--- SystemManager: All custom systems initialized successfully. ---\n")
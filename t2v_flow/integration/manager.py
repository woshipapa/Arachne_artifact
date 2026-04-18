# my_system/manager.py

import threading
from typing import List
from .base import CustomSystem  # 导入基类以进行类型检查

# --- 单例元类 ---
class SingletonMeta(type):
    """
    一个线程安全的单例元类，确保任何使用此元类的类都只有一个实例。
    """
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


# --- 系统管理器 ---
class SystemManager(metaclass=SingletonMeta):
    """
    一个单例的系统管理器，负责注册和初始化所有自定义系统。

    它充当了主程序与各个自定义模块之间的协调者。
    """
    def __init__(self):
        """
        构造函数只在第一次创建实例时被调用。
        """
        self._systems: List[CustomSystem] = []
        print("SystemManager Singleton initialized.")

    def register(self, system: CustomSystem):
        """
        注册一个新的自定义系统实例。

        Args:
            system (CustomSystem): 一个 CustomSystem 的子类实例。

        Raises:
            TypeError: 如果传入的对象不是 CustomSystem 的实例。
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
        按照注册的顺序，统一初始化所有已注册的系统。

        Args:
            args: 从上层框架传递的配置对象，将原样传递给每个系统的 initialize 方法。
        """
        print("\n--- SystemManager: Initializing all custom systems... ---")
        if not self._systems:
            print("No custom systems registered. Skipping.")
            return
        
        for system in self._systems:
            try:
                print(f"Initializing system: '{system.name}'...")
                # 调用每个系统自己的初始化逻辑
                system.initialize(args)
            except Exception as e:
                # 增加了健壮的错误处理，如果某个系统初始化失败，会打印明确的错误信息
                print(f"\nFATAL: Failed to initialize system '{system.name}'.")
                print(f"Error: {e}\n")
                # 重新引发异常，中断程序，防止在不完整的状态下继续运行
                raise
                
        print("--- SystemManager: All custom systems initialized successfully. ---\n")
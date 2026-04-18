# my_system/base.py

from abc import ABC, abstractmethod

class CustomSystem(ABC):
    """
    所有自定义系统的抽象基类 (Abstract Base Class)。

    它定义了一个统一的接口，所有希望被 SystemManager 管理的系统
    都必须继承自这个类，并实现其抽象方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        系统的唯一名称，主要用于日志记录和调试。
        每个子类都必须重写此属性。

        Returns:
            str: 系统的名称。
        """
        pass

    @abstractmethod
    def initialize(self, args):
        """
        系统的核心初始化逻辑。
        
        这个方法将在 Megatron 的分布式环境就绪后，由 SystemManager 统一调用。
        子类必须实现具体的初始化步骤。

        Args:
            args: 从上层框架（如 Megatron）传递过来的命令行参数或配置对象。
                  系统可以利用它来获取所需的配置信息。
        """
        pass
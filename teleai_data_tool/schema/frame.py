from dataclasses import dataclass, field
from typing import List, Optional


from .annotation import Instance


@dataclass
class Image:
    file_name: Optional[str] = None
    height: Optional[float] = None
    width: Optional[float] = None
    id: Optional[int] = None
    channel: int = 1  # 1 for gray and 3 for bgr


@dataclass
class Frame:
    image: Image
    id: Optional[int] = None
    instances: List[Instance] = field(default_factory=list)  # gt instances
    timestamp: Optional[float] = None

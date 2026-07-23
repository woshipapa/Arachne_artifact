from dataclasses import dataclass, field
from .frame import Frame, Image
from .annotation import Caption, FilterState, CameraMeta
from typing import List, Optional, Dict


@dataclass
class Clip:
    id: int
    file_path: str
    height: float
    width: float
    length: float
    fps: int
    valid_range: List[str] = field(default_factory=list)
    tags: List[str] = field(
        default_factory=list
    )
    # buildings, animation
    raw_video_path: str = ""
    start_frame_id: int = 0
    end_frame_id: int = 0
    frames: List[Frame] = field(default_factory=list)
    caption: Optional[Caption] = None
    filter_state: Optional[FilterState] = None
    camera_meta: Optional[CameraMeta] = None
    camera_movement: Optional[str] = (
        None
    )
    # tilt down, tilt left, tilt right, around left , around right, static shot, handheld shot
    meta: Dict[str, str] = field(default_factory=dict)  # other meta info
    valid_rect: List[int] = field(
        default_factory=list
    )

    def __post_init__(self):
        if self.filter_state is not None:
            self.filter_state.area = self.height * self.width

    @property
    def aspect_ratio(self):
        return self.height / self.width

    @property
    def num_frames(self):
        return self.end_frame_id - self.start_frame_id + 1


@dataclass
class ImageWithCaption:
    image: Image
    caption: Optional[Caption] = None
    filter_state: Optional[FilterState] = None
    meta: Dict[str, str] = field(default_factory=dict)  # other meta info

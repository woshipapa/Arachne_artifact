import numpy as np
from .data_client import DataClient
import os
from loguru import logger
from decord import VideoReader
from PIL import Image, ImageFile
from teleai_data_tool.utils.file import load_file

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FileClient(DataClient):
    def __init__(self, data_type: str = "video", path_spliter: str = ":") -> None:
        self.data_type = data_type
        self.path_spliter = path_spliter

    def get(self, data_info: str) -> np.ndarray:
        data_path, data_name = data_info.split(self.path_spliter)
        file_path = os.path.join(data_path, data_name)
        if self.data_type == "image":
            data = Image.open(file_path)
        elif self.data_type == "video":
            data = VideoReader(file_path)
        elif self.data_type == "dict":
            data = load_file(file_path)
        else:
            logger.error(f"data {self.data_type} is not supported")
        return data

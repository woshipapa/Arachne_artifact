import os
import shutil
from decord import VideoReader
from PIL import Image
from teleai_data_tool.utils import load_file, save_file
from .base_dataset import BaseDataset


class FileDataset(BaseDataset):
    def __init__(self, data_size, data_type, data_name=None, **kwargs):
        super(FileDataset, self).__init__(**kwargs)
        self.data_size = data_size
        self.data_type = data_type
        self.data_name = data_name
        self.data_dict = dict()

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        config_path = config.get("config_path", None)
        data_path = config.get("data_path", None)
        data_size = config["data_size"]
        data_type = config["data_type"]
        data_name = config["data_name"]
        return cls(
            config_path=config_path,
            data_path=data_path,
            data_size=data_size,
            data_type=data_type,
            data_name=data_name,
        )

    def open(self):
        pass
        # if len(self.data_dict) == 0:
        #     data_dir = os.path.join(self.data_path, 'data')
        #     data_paths = utils.list_dir(data_dir)
        #     if self.data_size is not None:
        #         assert len(data_paths) == self.data_size
        #     else:
        #         self.data_size = len(data_paths)
        #     for data_path in data_paths:
        #         data_name = os.path.basename(data_path)
        #         data_index = data_name.split('.')[0]
        #         assert data_index not in self.data_dict
        #         self.data_dict[data_index] = data_path

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        # data = self.data_dict[str(index)]
        if self.data_type == "image":
            data = os.path.join(self.data_path, "data", f"{index}.png")
            data = Image.open(data)
        elif self.data_type == "video":
            data = os.path.join(self.data_path, "data", f"{index}.mp4")
            data = VideoReader(data)
        elif self.data_type == "dict":
            data = os.path.join(self.data_path, "data", f"{index}.pkl")
            data = load_file(data)
        else:
            assert self.data_type == "raw"
        if self.data_name is not None:
            data_dict = {self.data_name: data}
        else:
            data_dict = data
        return data_dict


class FileWriter:
    def __init__(self, data_path, rmtree=True):
        if os.path.exists(data_path) and rmtree:
            shutil.rmtree(data_path)
        self.save_dir = os.path.join(data_path, "data")
        self.data_path = data_path
        self.data_type = None
        self.key_names = []
        self._count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def close(self):
        self.key_names = []
        self._count = 0

    def write_image(self, index, image, ext=None):
        if self.data_type is None:
            self.data_type = "image"
        else:
            assert self.data_type == "image"
        if isinstance(image, str):
            if ext is None:
                ext = image.split(".")[-1]
            save_path = os.path.join(self.save_dir, f"{index}.{ext}")
            os.system(f"cp {image} {save_path}")
        elif isinstance(image, bytes):
            if ext is None:
                ext = "png"
            save_path = os.path.join(self.save_dir, f"{index}.{ext}")
            open(save_path, "wb").write(image)
        elif isinstance(image, Image.Image):
            if ext is None:
                ext = "png"
            save_path = os.path.join(self.save_dir, f"{index}.{ext}")
            image.save(save_path)
        else:
            assert False
        self._count += 1

    def write_video(self, index, video, ext=None):
        if self.data_type is None:
            self.data_type = "video"
        else:
            assert self.data_type == "video"
        if isinstance(video, str):
            if ext is None:
                ext = video.split(".")[-1]
            save_path = os.path.join(self.save_dir, f"{index}.{ext}")
            os.system(f"cp {video} {save_path}")
        elif isinstance(video, bytes):
            if ext is None:
                ext = "mp4"
            save_path = os.path.join(self.save_dir, f"{index}.{ext}")
            open(save_path, "wb").write(video)
        else:
            assert False
        self._count += 1

    def write_dict(self, index, data):
        if self.data_type is None:
            self.data_type = "dict"
        else:
            assert self.data_type == "dict"
        assert isinstance(data, dict)
        self.key_names = list(set(self.key_names + list(data.keys())))
        save_path = os.path.join(self.save_dir, f"{index}.pkl")
        save_file(save_path, data)
        self._count += 1

    def write_config(self, **kwargs):
        config_path = os.path.join(self.data_path, "config.json")
        data_name = kwargs.pop("data_name", None)
        if data_name is not None:
            assert self.data_type != "dict"
        else:
            if self.data_type == "image":
                data_name = "image"
            elif self.data_type == "video":
                data_name = "video"
            elif self.data_type == "numpy":
                data_name = "data"
        if self.data_type == "dict":
            key_names = self.key_names
        else:
            key_names = [data_name]
        key_names.sort()
        config = {
            "_class_name": "FileDataset",
            "_key_names": key_names,
            "data_size": self._count,
            "data_type": self.data_type,
            "data_name": data_name,
        }
        config.update(kwargs)
        save_file(config_path, config)

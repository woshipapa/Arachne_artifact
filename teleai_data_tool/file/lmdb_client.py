import lmdb
import numpy as np
from .data_client import DataClient
import pickle
from io import BytesIO
from decord import VideoReader
from PIL import Image, ImageFile
from teleai_data_tool.datasets.lmdb_dataset import load_npy_from_stream

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_lmdb_keys(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        readonly=True,
        max_readers=1024,
        max_spare_txns=2,
        map_size=1024 * 1024 * 1024,
    )

    with env.begin() as txn:
        with txn.cursor() as cursor:
            keys = []
            for key in cursor.iternext(keys=True, values=False):
                keys.append(key)

    env.close()
    return keys


class LmdbClient(DataClient):
    def __init__(self, data_type: str = "video", path_spliter: str = ":") -> None:
        self.data_map = dict()
        self.data_type = data_type
        self.path_spliter = path_spliter

    def get(self, data_info: str) -> np.ndarray:
        data_path, data_id = data_info.split(self.path_spliter)
        if data_path not in self.data_map:
            data_env = lmdb.open(
                data_path, readonly=True, lock=False, readahead=False, meminit=False
            )
            data_txn = data_env.begin()
            self.data_map[data_path] = dict(env=data_env, txn=data_txn)
        data = self.data_map[data_path]["txn"].get(data_id.encode())
        if self.data_type == "image":
            data = Image.open(BytesIO(data))
        elif self.data_type == "video":
            data = VideoReader(BytesIO(data))
        elif self.data_type == "numpy":
            data = load_npy_from_stream(BytesIO(data))
        elif self.data_type == "dict":
            data = pickle.loads(data)
        return data

    def __del__(self):
        for _, v in self.data_map.items():
            v["env"].close()

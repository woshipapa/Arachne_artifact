import os


def get_root_dir():
    return os.path.abspath(__file__).split("vast")[0][:-1]


def get_data_dir():
    return os.environ.get("VAST_DATASETS_DIR", "./data/")

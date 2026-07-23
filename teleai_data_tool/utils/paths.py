import os
import fnmatch


def get_root_dir():
    return os.path.abspath(__file__).split("giga_datasets")[0][:-1]


def get_data_dir():
    return os.environ.get("GIGA_DATASETS_DIR", "./data/")


def find_files(directory, patterns):
    matched_files = set()
    for root, dirs, files in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                matched_files.add(os.path.join(root, filename))
    return list(matched_files)
